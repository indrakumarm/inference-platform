import contextlib
from copy import deepcopy
import io
import json
import subprocess
import unittest
from unittest.mock import Mock, patch

from inference_platform import cli, remote
from inference_platform.ssh import SSHHost
from inference_platform.ubuntu import plan


def report():
    def result(stdout=""):
        return {"code": 0, "stdout": stdout, "stderr": ""}
    return {
        "schema": 1,
        "os": {"ID": "ubuntu", "VERSION_ID": "22.04", "PRETTY_NAME": "Ubuntu 22.04"},
        "architecture": "x86_64", "root": False, "privileged": True,
        "root_free_bytes": 100 * 1024**3, "cache_free_bytes": 100 * 1024**3,
        "docker_installed": True, "toolkit_installed": True,
        "checks": {"gpu": result("0, NVIDIA A10G, 23028 MiB, 595.112.07"),
                   "gpu_processes": result(), "systemd": result(),
                   "docker_info": result("runc nvidia"), "containers": result(),
                   "docker_active": result("active")},
    }


class PlanningTests(unittest.TestCase):
    def test_existing_ready_runtime_needs_only_verification(self):
        self.assertEqual(plan(report()).actions, ["verify-gpu"])
        self.assertEqual(plan(report()).blockers, [])

    def test_missing_packages_get_ordered_plan_not_driver_install(self):
        facts = report()
        facts.update(docker_installed=False, toolkit_installed=False)
        facts["checks"]["docker_info"].update(code=127, stdout="")
        facts["checks"]["docker_active"].update(code=3, stdout="inactive")
        self.assertEqual(plan(facts).actions,
                         ["install-docker", "install-toolkit", "configure-runtime", "verify-gpu"])
        self.assertEqual(plan(facts).blockers, [])

    def test_blockers_prevent_unsupported_or_unsafe_preparation(self):
        changes = [
            lambda f: f["os"].update(ID="amzn"),
            lambda f: f.update(architecture="aarch64"),
            lambda f: f.update(privileged=False),
            lambda f: f.update(root_free_bytes=0),
            lambda f: f["checks"]["gpu"].update(code=1),
            lambda f: f["checks"]["gpu_processes"].update(code=1),
            lambda f: f["checks"]["gpu_processes"].update(stdout="1234 python"),
            lambda f: f["checks"]["systemd"].update(code=1),
        ]
        for change in changes:
            facts = report()
            change(facts)
            with self.subTest(change=change):
                self.assertTrue(plan(facts).blockers)

    def test_running_containers_block_runtime_changes_not_verification(self):
        facts = report()
        facts["checks"]["containers"]["stdout"] = "abc existing-service"
        self.assertFalse(plan(facts).blockers)
        facts["checks"]["docker_info"]["stdout"] = "runc"
        self.assertTrue(plan(facts).blockers)

    def test_active_unqueryable_docker_is_not_silently_restarted(self):
        facts = report()
        facts["checks"]["docker_info"].update(code=1, stdout="")
        self.assertTrue(plan(facts).blockers)
        self.assertNotIn("start-docker", plan(facts).actions)


class WorkflowTests(unittest.TestCase):
    def invoke(self, operation, facts=None, confirmation=None, terminal=True):
        host = Mock()
        host.inspect.return_value = facts or report()
        output = io.StringIO()
        with patch("inference_platform.cli.SSHHost", return_value=host), \
                patch("inference_platform.cli.sys.stdin.isatty", return_value=terminal), \
                patch("builtins.input", return_value=confirmation) as prompt, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = cli.main([operation, "--host", "ubuntu@gpu.example"])
        return code, host, prompt, output.getvalue()

    def test_inspect_is_read_only_even_when_changes_are_needed(self):
        facts = report()
        facts["toolkit_installed"] = False
        code, host, prompt, output = self.invoke("inspect", facts)
        self.assertEqual(code, 0)
        host.prepare.assert_not_called()
        prompt.assert_not_called()
        self.assertIn("install NVIDIA Container Toolkit", output)

    def test_cancel_or_noninteractive_never_prepares(self):
        for terminal, answer in ((True, "no"), (False, "prepare ubuntu@gpu.example")):
            with self.subTest(terminal=terminal):
                code, host, prompt, _ = self.invoke("prepare", confirmation=answer, terminal=terminal)
                self.assertEqual(code, 0 if terminal else 2)
                host.prepare.assert_not_called()
                if not terminal:
                    prompt.assert_not_called()

    def test_blockers_skip_confirmation_and_mutation(self):
        facts = report()
        facts["checks"]["gpu"]["code"] = 1
        code, host, prompt, _ = self.invoke("prepare", facts)
        self.assertEqual(code, 2)
        prompt.assert_not_called()
        host.prepare.assert_not_called()

    def test_confirmation_reinspects_and_verifies(self):
        code, host, _, output = self.invoke("prepare", confirmation="prepare ubuntu@gpu.example")
        self.assertEqual(code, 0)
        self.assertEqual(host.inspect.call_count, 3)
        host.prepare.assert_called_once_with(["verify-gpu"], False)
        self.assertIn("No model was deployed", output)

    def test_host_change_after_confirmation_aborts(self):
        original = report()
        changed = deepcopy(original)
        changed["checks"]["gpu_processes"]["stdout"] = "1234 another-job"
        host = Mock()
        host.inspect.side_effect = [original, changed]
        with patch("inference_platform.cli.SSHHost", return_value=host), \
                patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="prepare ubuntu@gpu.example"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["prepare", "--host", "ubuntu@gpu.example"]), 2)
        host.prepare.assert_not_called()

    def test_preparation_failure_is_not_reported_as_ready(self):
        host = Mock()
        host.inspect.return_value = report()
        host.prepare.side_effect = subprocess.CalledProcessError(1, ["ssh"])
        output = io.StringIO()
        with patch("inference_platform.cli.SSHHost", return_value=host), \
                patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="prepare ubuntu@gpu.example"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(cli.main(["prepare", "--host", "ubuntu@gpu.example"]), 1)
        self.assertNotIn("Host runtime ready:", output.getvalue())

    def test_incomplete_preparation_fails_post_check(self):
        original = report()
        missing_runtime = deepcopy(original)
        missing_runtime["checks"]["docker_info"]["stdout"] = "runc"
        host = Mock()
        host.inspect.side_effect = [original, original, missing_runtime]
        output = io.StringIO()
        with patch("inference_platform.cli.SSHHost", return_value=host), \
                patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="prepare ubuntu@gpu.example"), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            self.assertEqual(cli.main(["prepare", "--host", "ubuntu@gpu.example"]), 2)
        self.assertIn("Post-preparation inspection failed", output.getvalue())
        self.assertNotIn("Host runtime ready:", output.getvalue())


class TransportAndRemoteTests(unittest.TestCase):
    def test_transport_checks_host_keys_and_sends_payload_without_shell(self):
        with patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = json.dumps(report())
            host = SSHHost("ubuntu@gpu.example")
            self.assertEqual(host.inspect(), report())
            command = run.call_args.args[0]
            self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertEqual(command[-1],
                             "env INFERENCE_PLATFORM_LOG_LEVEL=WARNING python3 - inspect")
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertIn("def inspect()", run.call_args.kwargs["input"])
            self.assertIn("def configure_logging()", run.call_args.kwargs["input"])
            host.prepare(["configure-runtime", "verify-gpu"], root=False)
            self.assertEqual(run.call_args.args[0][-1],
                             "sudo -n -- env INFERENCE_PLATFORM_LOG_LEVEL=WARNING "
                             "python3 - prepare configure-runtime verify-gpu")

    def test_remote_logging_uses_configured_level(self):
        with patch.dict("os.environ", {"INFERENCE_PLATFORM_LOG_LEVEL": "DEBUG"}), \
                patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = json.dumps(report())
            host = SSHHost("ubuntu@gpu.example")
            host.inspect()
        self.assertIn("INFERENCE_PLATFORM_LOG_LEVEL=DEBUG", run.call_args.args[0][-1])

    def test_invalid_targets_and_report_formats_fail_explicitly(self):
        for target in ("-oProxyCommand=bad", "user@host;touch /tmp/x", "host", "user@host x"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                SSHHost(target)
        for text in ("login banner\n{}", "[]", '{"schema": 2}'):
            with patch("inference_platform.ssh.subprocess.run", return_value=Mock(stdout=text)):
                with self.assertRaises(ValueError):
                    SSHHost("ubuntu@gpu.example").inspect()

    def test_remote_probe_missing_command_is_explicit(self):
        with patch("inference_platform.remote.subprocess.run", side_effect=FileNotFoundError("missing")):
            result = remote.probe(["nvidia-smi"])
        self.assertEqual(result["code"], 127)
        self.assertIn("missing", result["stderr"])

    def test_inspection_collects_only_probes_without_preparing(self):
        with patch("inference_platform.remote.os.geteuid", return_value=0), \
                patch("inference_platform.remote.os_release",
                      return_value={"ID": "ubuntu", "VERSION_ID": "22.04"}), \
                patch("inference_platform.remote.probe", return_value={
                    "code": 0, "stdout": "", "stderr": ""}) as probe, \
                patch("inference_platform.remote.run") as mutate:
            facts = remote.inspect()
        mutate.assert_not_called()
        commands = [call.args[0] for call in probe.call_args_list]
        self.assertTrue(all(c[0] in ("nvidia-smi", "test", "docker", "systemctl") for c in commands))
        self.assertFalse(any("run" in c or "restart" in c or "install" in c for c in commands))
        self.assertEqual(facts["schema"], 1)

    def test_remote_rechecks_workloads_before_mutations(self):
        busy = {"code": 0, "stdout": "1234 python", "stderr": ""}
        with patch("inference_platform.remote.probe", return_value=busy):
            with self.assertRaisesRegex(RuntimeError, "GPU processes"):
                remote.ensure_idle()

    def test_remote_verification_failure_propagates(self):
        with patch("inference_platform.remote.os.geteuid", return_value=0), \
                patch("inference_platform.remote.os_release",
                      return_value={"ID": "ubuntu", "VERSION_ID": "22.04"}), \
                patch("inference_platform.remote.os.uname", return_value=Mock(machine="x86_64")), \
                patch("inference_platform.remote.run",
                      side_effect=subprocess.CalledProcessError(1, ["docker"])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                remote.prepare(["verify-gpu"])
            self.assertIn("--rm", run.call_args.args[0])
            self.assertIn("--gpus", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
