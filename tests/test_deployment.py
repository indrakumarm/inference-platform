import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request
from unittest.mock import Mock, patch

from inference_platform import cli, remote_deploy, remote_monitoring
from inference_platform.deployment import deployment_blockers, parse_profile
from inference_platform.providers.ssh import SSHProvider
from inference_platform.ssh import SSHHost
from inference_platform.ubuntu import plan


def profile_data():
    return {
        "name": "small-model",
        "image": "vllm/vllm-openai@sha256:" + "a" * 64,
        "model": "org/model",
        "model_revision": "b" * 40,
        "tokenizer": "org/model",
        "tokenizer_revision": "b" * 40,
        "gpu_indices": [0],
        "min_gpu_memory_gib": 16,
        "min_free_disk_gib": 40,
        "max_model_len": 4096,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "port": 8000,
        "shm_size_gib": 8,
    }


def deployment_facts():
    return {
        "docker_ready": True, "docker_error": "",
        "gpu_error": "", "gpus": [
            {"index": 0, "name": "NVIDIA A10G",
             "total_memory_mib": 23028, "free_memory_mib": 23028},
        ],
        "gpu_process_error": "", "gpu_processes": [],
        "container_error": "", "running_containers": [],
        "target_container_exists": False,
        "free_disk_bytes": 100 * 1024**3, "disk_error": "",
        "port_available": True,
    }


def host_report():
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


def fake_provider():
    """A mocked provider that still applies the real planner and preflight policy."""
    host = Mock()
    host.plan.side_effect = plan
    host.deployment_blockers.side_effect = deployment_blockers
    return host


class ProfileTests(unittest.TestCase):
    def test_private_binding_is_optional_and_rejects_non_private_addresses(self):
        self.assertIsNone(parse_profile(profile_data()).private_bind_ip)
        for address in ("0.0.0.0", "127.0.0.1", "8.8.8.8", "::1", "gpu.internal"):
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "private_bind_ip"):
                parse_profile({**profile_data(), "private_bind_ip": address})

    def test_preflight_reports_private_binding_failure(self):
        data = {**profile_data(), "private_bind_ip": "10.0.1.10"}
        facts = {**deployment_facts(), "private_bind_error": "address not assigned"}
        expected = ["Cannot bind 10.0.1.10:8000: address not assigned"]
        self.assertEqual(deployment_blockers(facts, parse_profile(data)), expected)
        self.assertEqual(remote_deploy.blockers(facts, data), expected)

    def test_preflight_reports_available_and_required_disk_space(self):
        profile = parse_profile(profile_data())
        facts = deployment_facts()
        facts["free_disk_bytes"] = int(38.25 * 1024**3)
        self.assertEqual(deployment_blockers(facts, profile), [
            "Free space on /: 38.25 GiB available; 40 GiB required (min_free_disk_gib).",
        ])
        facts["free_disk_bytes"] = 40 * 1024**3
        self.assertEqual(deployment_blockers(facts, profile), [])

    def test_requires_immutable_image_and_model_revisions(self):
        data = profile_data()
        self.assertEqual(parse_profile(data).gpu_indices, [0])
        for field, value in (("image", "vllm/vllm-openai:latest"),
                             ("model_revision", "main"),
                             ("tokenizer_revision", "f" * 39)):
            invalid = profile_data()
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_profile(invalid)

    def test_rejects_duplicate_gpu_assignments_and_unknown_options(self):
        invalid = profile_data()
        invalid["gpu_indices"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "gpu_indices"):
            parse_profile(invalid)
        invalid = profile_data()
        invalid["extra_option"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            parse_profile(invalid)

    def test_preflight_blocks_workloads_memory_disk_and_public_port(self):
        profile = parse_profile(profile_data())
        facts = deployment_facts()
        self.assertEqual(deployment_blockers(facts, profile), [])
        facts["gpu_processes"] = ["123 python"]
        facts["running_containers"] = ["other-service"]
        facts["gpus"][0]["free_memory_mib"] = 1
        facts["free_disk_bytes"] = 0
        facts["port_available"] = False
        blockers = deployment_blockers(facts, profile)
        self.assertEqual(len(blockers), 5)
        facts = deployment_facts()
        facts["target_container_exists"] = True
        self.assertIn("already exists", " ".join(deployment_blockers(facts, profile)))


class DeploymentWorkflowTests(unittest.TestCase):
    def test_file_discovery_updates_follow_successful_lifecycle_actions(self):
        data = {**profile_data(), "private_bind_ip": "10.0.1.10"}
        stopped = {"name": "inference-small-model", "exists": True, "state": "exited",
                   "running": False, "exit_code": 0, "error": "", "finished_at": "now"}
        for operation, fail_deploy, absent in (
            ("deploy", False, False), ("deploy", True, False),
            ("remove", False, False), ("remove", False, True),
        ):
            with self.subTest(operation=operation, fail_deploy=fail_deploy, absent=absent), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profile.json"
                path.write_text(json.dumps(data))
                workflow = Mock()
                gpu, monitor = workflow.gpu, workflow.monitor
                gpu.plan.side_effect = plan
                gpu.deployment_blockers.side_effect = deployment_blockers
                gpu.inspect.return_value = host_report()
                gpu.deployment_inspect.return_value = {**deployment_facts(), "private_bind_error": ""}
                gpu.deploy.return_value = {"model": "org/model", "endpoint": "http://127.0.0.1:8000/v1"}
                if fail_deploy:
                    gpu.deploy.side_effect = RuntimeError("model not ready")
                status = {**stopped, "exists": False} if absent else stopped
                gpu.lifecycle.side_effect = [status, status, {"name": stopped["name"]}]
                answer = "deploy ubuntu@gpu.example small-model" if operation == "deploy" else "yes"
                with patch("inference_platform.cli.SSHHost", return_value=monitor), \
                        patch("inference_platform.cli.create_provider", return_value=gpu), \
                        patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                        patch("builtins.input", return_value=answer), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    result = cli.main([operation, "--host", "ubuntu@gpu.example",
                                       "--profile", str(path), "--monitoring-host", "metrics@monitor.example"])
                self.assertEqual(result, int(fail_deploy))
                if fail_deploy:
                    monitor.update_monitoring_target.assert_not_called()
                else:
                    monitor.update_monitoring_target.assert_called_once_with(
                        "add" if operation == "deploy" else "remove", "10.0.1.10", 8000, "small-model")
                    preceding = ("gpu.deploy" if operation == "deploy" else "gpu.lifecycle")
                    self.assertEqual([call[0] for call in workflow.mock_calls][-2:],
                                     [preceding, "monitor.update_monitoring_target"])
                    if operation == "remove":
                        self.assertEqual(gpu.lifecycle.call_count, 2 if absent else 3)

    def run_cli(self, confirmation):
        host = fake_provider()
        host.inspect.return_value = host_report()
        host.deployment_inspect.return_value = deployment_facts()
        host.deploy.return_value = {
            "endpoint": "http://127.0.0.1:8000/v1", "model": "org/model",
            "generation_smoke_test": "passed",
        }
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_data()))
            with patch("inference_platform.cli.create_provider", return_value=host), \
                    patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value=confirmation), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = cli.main(["deploy", "--host", "ubuntu@gpu.example",
                                 "--profile", str(path)])
        return code, host, output.getvalue()

    def test_deploy_requires_confirmation_then_reports_readiness(self):
        code, host, output = self.run_cli("deploy ubuntu@gpu.example small-model")
        self.assertEqual(code, 0)
        host.deploy.assert_called_once()
        self.assertEqual(host.inspect.call_count, 2)
        self.assertEqual(host.deployment_inspect.call_count, 2)
        self.assertIn("generation smoke test passed", output)
        self.assertIn("127.0.0.1:8000", output)

    def test_cancelled_deployment_does_not_mutate_host(self):
        code, host, output = self.run_cli("no")
        self.assertEqual(code, 0)
        host.deploy.assert_not_called()
        self.assertIn("Cancelled; nothing changed.", output)

    def test_new_gpu_workload_after_confirmation_blocks_deployment(self):
        host = fake_provider()
        host.inspect.return_value = host_report()
        busy = deployment_facts()
        busy["gpu_processes"] = ["123 python"]
        host.deployment_inspect.side_effect = [deployment_facts(), busy]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_data()))
            with patch("inference_platform.cli.create_provider", return_value=host), \
                    patch("inference_platform.cli.sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value="deploy ubuntu@gpu.example small-model"), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = cli.main(["deploy", "--host", "ubuntu@gpu.example",
                                 "--profile", str(path)])
        self.assertEqual(code, 2)
        host.deploy.assert_not_called()
        self.assertIn("GPU compute processes are active", output.getvalue())

    def test_unprepared_host_is_not_deployed(self):
        host = fake_provider()
        host.inspect.return_value = host_report()
        host.inspect.return_value["checks"]["docker_info"]["stdout"] = "runc"
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_data()))
            with patch("inference_platform.cli.create_provider", return_value=host), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = cli.main(["deploy", "--host", "ubuntu@gpu.example",
                                 "--profile", str(path)])
        self.assertEqual(code, 2)
        host.deployment_inspect.assert_not_called()

    def test_list_reads_remote_inventory_without_inspecting_host_or_profile(self):
        deployments = [
            {"container": "inference-small-model", "profile": "small-model",
             "model": "org/model", "state": "running", "status": "Up",
             "image": "vllm/vllm-openai@sha256:abc", "ports": "127.0.0.1:8000->8000/tcp",
             "metadata_source": "labels"},
        ]
        host = fake_provider()
        host.deployment_list.return_value = deployments
        output = io.StringIO()
        with patch("inference_platform.cli.create_provider", return_value=host), \
                contextlib.redirect_stdout(output):
            code = cli.main(["list", "--host", "ubuntu@gpu.example"])
        self.assertEqual(code, 0)
        host.deployment_list.assert_called_once()
        host.inspect.assert_not_called()
        self.assertIn("inference-small-model [running]", output.getvalue())

    def run_lifecycle_cli(self, operation, statuses, answer=None, terminal=True):
        host = fake_provider()
        host.lifecycle.side_effect = statuses
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(profile_data()))
            with patch("inference_platform.cli.create_provider", return_value=host), \
                    patch("inference_platform.cli.sys.stdin.isatty", return_value=terminal), \
                    patch("builtins.input", return_value=answer) as prompt, \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = cli.main([operation, "--host", "ubuntu@gpu.example",
                                 "--profile", str(path)])
        return code, host, prompt, output.getvalue()

    def test_status_is_read_only_and_does_not_prompt(self):
        running = {"name": "inference-small-model", "exists": True,
                   "state": "running", "running": True, "started_at": "now"}
        code, host, prompt, output = self.run_lifecycle_cli("status", [running])
        self.assertEqual(code, 0)
        prompt.assert_not_called()
        self.assertEqual(host.lifecycle.call_args_list[0].args[0], "status")
        self.assertIn("State: running", output)

    def test_stop_accepts_yes_and_rechecks_state(self):
        running = {"name": "inference-small-model", "exists": True,
                   "state": "running", "running": True, "started_at": "now",
                   "exit_code": 0, "error": "", "finished_at": ""}
        stopped = {"name": "inference-small-model", "exists": True,
                   "state": "exited", "running": False, "exit_code": 0,
                   "error": "", "finished_at": "now"}
        code, host, prompt, output = self.run_lifecycle_cli(
            "stop", [running, running, stopped], answer=" YES ")
        self.assertEqual(code, 0)
        prompt.assert_called_once_with("Stop inference-small-model on ubuntu@gpu.example? [y/N]: ")
        self.assertEqual([call.args[0] for call in host.lifecycle.call_args_list],
                         ["status", "status", "stop"])
        self.assertIn("State: exited", output)

    def test_start_cancels_without_affirmative_confirmation(self):
        stopped = {"name": "inference-small-model", "exists": True,
                   "state": "exited", "running": False, "exit_code": 0,
                   "error": "", "finished_at": "now"}
        for answer in ("no", "", "maybe"):
            with self.subTest(answer=answer):
                code, host, prompt, output = self.run_lifecycle_cli(
                    "start", [stopped], answer=answer)
                self.assertEqual(code, 0)
                prompt.assert_called_once_with(
                    "Start inference-small-model on ubuntu@gpu.example? [y/N]: ")
                self.assertEqual([call.args[0] for call in host.lifecycle.call_args_list],
                                 ["status"])
                self.assertIn("Cancelled; nothing changed.", output)

    def test_remove_refuses_running_container_without_confirmation(self):
        running = {"name": "inference-small-model", "exists": True,
                   "state": "running", "running": True, "started_at": "now",
                   "exit_code": 0, "error": "", "finished_at": ""}
        code, host, prompt, output = self.run_lifecycle_cli("remove", [running])
        self.assertEqual(code, 2)
        prompt.assert_not_called()
        self.assertEqual(host.lifecycle.call_count, 1)
        self.assertIn("Stop the container before removing it", output)


class MonitoringTargetTests(unittest.TestCase):
    def test_multiple_targets_can_be_registered_and_removed_independently(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(remote_monitoring, "TARGET_DIR", Path(directory)):
            first = {"name": "small-model", "ip": "10.0.1.10", "port": 8000}
            second = {**first, "ip": "10.0.1.11"}
            for target in (first, first, second):
                remote_monitoring.update_target("add", target)
            self.assertEqual(len(list(Path(directory).iterdir())), 2)
            remote_monitoring.update_target("remove", first)
            remote_monitoring.update_target("remove", first)
            remaining, = Path(directory).glob("vllm-*.json")
            self.assertEqual(json.loads(remaining.read_text()), [
                {"targets": ["10.0.1.11:8000"], "labels": {"deployment": "small-model"}},
            ])


class DeploymentTransportTests(unittest.TestCase):
    def test_payload_and_profile_are_sent_over_host_key_checked_ssh(self):
        with patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = json.dumps(deployment_facts())
            host = SSHProvider(SSHHost("ubuntu@gpu.example"))
            result = host.deployment_inspect(profile_data())
        self.assertTrue(result["docker_ready"])
        command = run.call_args.args[0]
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("def deployment_inspect", command[-1])
        self.assertEqual(run.call_args.kwargs["input"], json.dumps(profile_data()))
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_deploy_streams_remote_diagnostics_while_capturing_json(self):
        with patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = json.dumps({
                "endpoint": "http://127.0.0.1:8000/v1",
                "model": "org/model",
                "generation_smoke_test": "passed",
            })
            host = SSHProvider(SSHHost("ubuntu@gpu.example"))
            self.assertEqual(host.deploy(profile_data())["model"], "org/model")
        self.assertEqual(run.call_args.kwargs["stdout"], subprocess.PIPE)
        self.assertNotIn("stderr", run.call_args.kwargs)
        self.assertNotIn("capture_output", run.call_args.kwargs)

    def test_list_sends_inventory_request_without_a_profile(self):
        with patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = "[]"
            host = SSHProvider(SSHHost("ubuntu@gpu.example"))
            self.assertEqual(host.deployment_list(), [])
        self.assertEqual(run.call_args.kwargs["input"], "{}")
        self.assertIn("list", run.call_args.args[0][-1])

    def test_ssh_failure_message_hides_embedded_remote_command(self):
        error = subprocess.CalledProcessError(
            255, ["ssh", "very-large-embedded-remote-script"],
            stderr="ssh: connect to host gpu.example port 22: Operation timed out",
        )
        with patch("inference_platform.ssh.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "SSH connection/authentication failed") as raised:
                SSHProvider(SSHHost("ubuntu@gpu.example")).deployment_list()
        self.assertIn("Operation timed out", str(raised.exception))
        self.assertNotIn("very-large-embedded-remote-script", str(raised.exception))

    def test_ssh_timeout_has_operation_context_without_command_dump(self):
        error = subprocess.TimeoutExpired(["ssh", "very-large-embedded-remote-script"], timeout=240)
        with patch("inference_platform.ssh.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "list on ubuntu@gpu.example timed out"):
                SSHProvider(SSHHost("ubuntu@gpu.example")).deployment_list()


class RemoteDeploymentTests(unittest.TestCase):
    def test_launch_adds_the_private_binding_without_replacing_loopback(self):
        profile = {**profile_data(), "private_bind_ip": "10.0.1.10"}
        with patch.object(remote_deploy, "deployment_inspect", return_value={
            **deployment_facts(), "private_bind_error": "",
        }), patch.object(remote_deploy, "docker_prefix", return_value=[]), \
                patch.object(remote_deploy, "run") as run, \
                patch.object(remote_deploy, "wait_for_model"):
            remote_deploy.deploy(profile)
        launch = next(command.args[0] for command in run.call_args_list if "-d" in command.args[0])
        bindings = [launch[index + 1] for index, value in enumerate(launch) if value == "--publish"]
        self.assertEqual(bindings, ["127.0.0.1:8000:8000", "10.0.1.10:8000:8000"])

    def test_launch_binds_only_loopback_and_uses_selected_gpus_and_pins(self):
        commands = []
        with patch("inference_platform.remote_deploy.deployment_inspect",
                   return_value=deployment_facts()), \
                patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.run",
                      side_effect=lambda command, **kwargs: commands.append(command)), \
                patch("inference_platform.remote_deploy.wait_for_model",
                      return_value={"endpoint": "http://127.0.0.1:8000/v1",
                                    "model": "org/model",
                                    "generation_smoke_test": "passed"}):
            result = remote_deploy.deploy(profile_data())
        launch = next(command for command in commands if "run" in command and "-d" in command)
        self.assertEqual(launch.count("--publish"), 1)
        self.assertIn("127.0.0.1:8000:8000", launch)
        self.assertIn("device=0", launch)
        self.assertIn(profile_data()["image"], launch)
        self.assertIn(profile_data()["model_revision"], launch)
        self.assertIn("com.inference-platform.managed=true", launch)
        self.assertIn("com.inference-platform.model=org/model", launch)
        image_index = launch.index(profile_data()["image"])
        self.assertEqual(launch[image_index + 1:image_index + 3],
                         ["--model", profile_data()["model"]])
        self.assertNotIn("serve", launch[image_index + 1:])
        self.assertEqual(result["generation_smoke_test"], "passed")

    def test_readiness_checks_identity_and_nonempty_generation(self):
        profile = profile_data()

        class Response:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(self.value).encode()

        responses = [
            Response({"data": [{"id": profile["model"]}]}),
            Response({"choices": [{"text": " ready"}]}),
        ]
        with patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.probe",
                      return_value={"code": 0, "stdout": "true", "stderr": ""}), \
                patch("inference_platform.remote_deploy.urllib.request.urlopen",
                      side_effect=responses) as urlopen:
            result = remote_deploy.wait_for_model(profile, "inference-small-model")
        self.assertEqual(result["model"], profile["model"])
        self.assertEqual(result["generation_smoke_test"], "passed")
        self.assertIsInstance(urlopen.call_args_list[1].args[0], Request)

    def test_identity_mismatch_fails_readiness(self):
        profile = profile_data()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"data":[{"id":"wrong-model"}]}'

        with patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.probe",
                      return_value={"code": 0, "stdout": "true", "stderr": ""}), \
                patch("inference_platform.remote_deploy.urllib.request.urlopen",
                      return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                remote_deploy.wait_for_model(profile, "inference-small-model")

    def test_container_status_reports_missing_and_inspects_only_target(self):
        with patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.probe",
                      return_value={"code": 0, "stdout": "other-container", "stderr": ""}):
            status = remote_deploy.container_status(profile_data())
        self.assertEqual(status, {"name": "inference-small-model",
                                  "exists": False, "state": "not-found"})
        state = {"Status": "exited", "Running": False, "ExitCode": 1,
                 "Error": "", "StartedAt": "then", "FinishedAt": "now"}
        with patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.probe", side_effect=[
                    {"code": 0, "stdout": "inference-small-model", "stderr": ""},
                    {"code": 0, "stdout": json.dumps(state), "stderr": ""},
                ]):
            status = remote_deploy.container_status(profile_data())
        self.assertEqual(status["state"], "exited")
        self.assertEqual(status["exit_code"], 1)

    def test_list_includes_labeled_and_legacy_inference_containers_only(self):
        rows = [
            {"Names": "database", "Labels": "", "State": "running",
             "Status": "Up", "Image": "postgres", "Ports": ""},
            {"Names": "inference-old-model", "Labels": "", "State": "exited",
             "Status": "Exited (1)", "Image": "vllm:v0", "Ports": ""},
            {"Names": "custom-name", "Labels":
             "com.inference-platform.managed=true,com.inference-platform.profile=large,"
             "com.inference-platform.model=org/model,com.inference-platform.port=8001",
             "State": "running", "Status": "Up", "Image": "vllm@sha256:abc",
             "Ports": "127.0.0.1:8001->8000/tcp"},
        ]
        output = "\n".join(json.dumps(row) for row in rows)
        with patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.probe",
                      return_value={"code": 0, "stdout": output, "stderr": ""}):
            deployments = remote_deploy.list_deployments()
        self.assertEqual([item["container"] for item in deployments],
                         ["custom-name", "inference-old-model"])
        legacy = deployments[1]
        self.assertEqual(legacy["model"], "unknown (legacy container)")
        self.assertEqual(legacy["metadata_source"], "container-name")

    def test_start_reports_disk_shortfall_without_starting_container(self):
        stopped = {"name": "inference-small-model", "exists": True,
                   "state": "exited", "running": False}
        facts = deployment_facts()
        facts["target_container_exists"] = True
        facts["free_disk_bytes"] = int(38.25 * 1024**3)
        with patch("inference_platform.remote_deploy.container_status", return_value=stopped), \
                patch("inference_platform.remote_deploy.deployment_inspect", return_value=facts), \
                patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.run") as run:
            with self.assertRaises(RuntimeError) as error:
                remote_deploy.manage_container("start", profile_data())
        self.assertEqual(str(error.exception), "Cannot start container: "
                         "Free space on /: 38.25 GiB available; 40 GiB required (min_free_disk_gib).")
        run.assert_not_called()

    def test_start_rechecks_dedicated_host_and_removal_requires_stopped(self):
        stopped = {"name": "inference-small-model", "exists": True,
                   "state": "exited", "running": False}
        running = {"name": "inference-small-model", "exists": True,
                   "state": "running", "running": True}
        with patch("inference_platform.remote_deploy.container_status",
                   side_effect=[stopped, running]), \
                patch("inference_platform.remote_deploy.deployment_inspect",
                      return_value=deployment_facts()) as inspect, \
                patch("inference_platform.remote_deploy.blockers", return_value=[]), \
                patch("inference_platform.remote_deploy.docker_prefix", return_value=[]), \
                patch("inference_platform.remote_deploy.run") as run:
            result = remote_deploy.manage_container("start", profile_data())
        inspect.assert_called_once()
        self.assertEqual(run.call_args.args[0][-2:],
                         ["start", "inference-small-model"])
        self.assertTrue(result["running"])

        with patch("inference_platform.remote_deploy.container_status", return_value=running), \
                patch("inference_platform.remote_deploy.run") as run:
            with self.assertRaisesRegex(RuntimeError, "stop it before removing"):
                remote_deploy.manage_container("remove", profile_data())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
