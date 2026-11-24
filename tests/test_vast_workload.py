import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from inference_platform import cli, remote_vast
from inference_platform.deployment import parse_profile
from inference_platform.providers import create_provider, parse_target
from inference_platform.providers.vast import VastProvider
from test_deployment import profile_data


def facts():
    return {
        "gpu_error": "", "torch_error": "", "gpu_process_error": "", "gpu_processes": [],
        "gpus": [{"index": 0, "name": "GPU", "free_memory_mib": 24 * 1024}],
        "free_disk_bytes": 100 * 1024**3, "disk_error": "",
        "shm_free_bytes": 8 * 1024**3, "shm_error": "",
        "port_available": True, "private_bind_error": "", "target_exists": False,
    }


def host_report():
    good = {"code": 0, "stdout": "", "stderr": ""}
    return {
        "schema": 1, "runtime": "vast-container", "os": {"PRETTY_NAME": "Vast container"},
        "architecture": "x86_64", "root": True, "privileged": False,
        "root_free_bytes": 100 * 1024**3, "cache_free_bytes": 100 * 1024**3,
        "checks": {"gpu": {**good, "stdout": "GPU"}, "gpu_processes": good,
                   "torch": {**good, "stdout": '{"available": true, "count": 1}'},
                   "vllm": {**good, "stdout": "/usr/bin/vllm"},
                   "cache": {**good, "stdout": "/root/.cache/huggingface/hub"}},
    }


def saved(profile, phase="running", **fields):
    return {"profile": profile, "phase": phase, "token": "token",
            "epoch": ["boot", "1"], "pid": 42, "starttime": "123", **fields}


class VastWorkloadTests(unittest.TestCase):
    def test_registered_target_uses_strict_ssh_transport(self):
        provider = create_provider(parse_target({"provider": "vastai", "host": "root@gpu.example", "port": 2345}))
        self.assertIsInstance(provider, VastProvider)
        self.assertIn("StrictHostKeyChecking=yes", provider.transport.command)
        self.assertIn("2345", provider.transport.command)

    def test_remote_payload_runs_without_installed_package(self):
        class LocalTransport:
            target = "root@example"
            log_level = "WARNING"

            def run(self, _operation, command, payload, timeout, **options):
                return subprocess.run(shlex.split(command), input=payload, text=True,
                                      check=True, timeout=timeout, **options)

        report = VastProvider(LocalTransport()).inspect()
        self.assertEqual(report["runtime"], "vast-container")
        self.assertIn("vllm", report["checks"])

    def test_plan_requires_gpu_torch_and_vllm_but_no_system_packages(self):
        provider = VastProvider(Mock())
        report = host_report()
        self.assertEqual(provider.plan(report).actions, [])
        self.assertEqual(provider.plan(report).blockers, [])
        report["checks"]["torch"]["stdout"] = '{"available": false, "count": 0}'
        self.assertTrue(provider.plan(report).blockers)
        report["checks"]["vllm"]["code"] = 127
        self.assertGreaterEqual(len(provider.plan(report).blockers), 2)

    def test_preflight_enforces_memory_shared_memory_and_loopback(self):
        profile = profile_data()
        minimum = {**facts(), "gpus": [{"index": 0, "name": "GPU", "free_memory_mib": 16 * 1024}],
                   "free_disk_bytes": 40 * 1024**3}
        self.assertEqual(remote_vast.blockers(minimum, profile), [])
        too_small = {**minimum, "gpus": [{"index": 0, "name": "GPU",
                                          "free_memory_mib": 16 * 1024 - 1}],
                     "free_disk_bytes": 40 * 1024**3 - 1}
        self.assertEqual(len(remote_vast.blockers(too_small, profile)), 2)
        bad = {**facts(), "shm_free_bytes": 1024, "port_available": False,
               "gpu_processes": ["123 python"]}
        blocked = remote_vast.blockers(bad, {**profile, "private_bind_ip": "10.0.0.1"})
        self.assertTrue(any("loopback" in item for item in blocked))
        self.assertTrue(any("/dev/shm" in item for item in blocked))
        self.assertTrue(any("Port" in item for item in blocked))
        self.assertTrue(any("GPU compute" in item for item in blocked))

    def test_cache_disk_checks_nearest_existing_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(remote_vast.existing_ancestor(root / "not-yet" / "hub"), root)

    def test_runtime_state_after_restart_and_lost_supervisor(self):
        state = saved(profile_data())
        with patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                patch.object(remote_vast, "supervisor_alive", return_value=False):
            self.assertEqual(remote_vast.state_status(state)["state"], "unknown")
        with patch.object(remote_vast, "container_epoch", return_value=["boot", "2"]):
            self.assertEqual(remote_vast.state_status(state)["state"], "exited")
        with patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                patch.object(remote_vast.time, "time", return_value=100):
            self.assertEqual(remote_vast.state_status(
                saved(profile_data(), phase="starting", requested_at=60))["state"], "unknown")

    def test_pid_reuse_is_not_accepted_as_supervisor(self):
        fields = ["S"] + ["0"] * 18 + ["different"]
        with patch.object(remote_vast, "proc_fields", return_value=fields), \
                patch.object(Path, "read_bytes", return_value=b"python\0token\0"):
            self.assertFalse(remote_vast.supervisor_alive(saved(profile_data())))

    def test_changed_profile_cannot_control_existing_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small-model"
            path.mkdir()
            remote_vast.write_state(path, saved(profile_data(), phase="exited"))
            changed = {**profile_data(), "model": "other/model"}
            with patch.object(remote_vast, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    remote_vast.status(changed)
                with self.assertRaisesRegex(RuntimeError, "original"):
                    remote_vast.manage("remove", changed)

    def test_unknown_process_is_not_removed_or_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small-model"
            path.mkdir()
            remote_vast.write_state(path, saved(profile_data()))
            with patch.object(remote_vast, "ROOT", root), \
                    patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                    patch.object(remote_vast, "supervisor_alive", return_value=False), \
                    patch.object(remote_vast.os, "kill") as kill:
                for operation in ("stop", "remove"):
                    with self.assertRaisesRegex(RuntimeError, "unknown"):
                        remote_vast.manage(operation, profile_data())
                kill.assert_not_called()
                self.assertTrue((path / "state.json").exists())

    def test_stop_rechecks_identity_after_opening_pidfd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small-model"
            path.mkdir()
            remote_vast.write_state(path, saved(profile_data()))
            with patch.object(remote_vast, "ROOT", root), \
                    patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                    patch.object(remote_vast, "supervisor_alive", side_effect=[True, False]), \
                    patch.object(remote_vast.os, "pidfd_open",
                                 side_effect=lambda _pid: os.open(os.devnull, os.O_RDONLY), create=True), \
                    patch.object(remote_vast.signal, "pidfd_send_signal", create=True) as send:
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    remote_vast.manage("stop", profile_data())
                send.assert_not_called()

    def test_other_active_deployment_blocks_concurrent_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root / "other"
            other.mkdir()
            remote_vast.write_state(other, saved({**profile_data(), "name": "other"}))
            with patch.object(remote_vast, "ROOT", root), \
                    patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                    patch.object(remote_vast, "supervisor_alive", return_value=True), \
                    patch.object(remote_vast, "require_ready"), \
                    patch.object(remote_vast.os, "posix_spawn") as spawn:
                with self.assertRaisesRegex(RuntimeError, "Another deployment"):
                    remote_vast.start_process(profile_data(), "source", create=True)
                spawn.assert_not_called()

    def test_deploy_cli_uses_vast_readiness_and_warns_image_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            profile = root / "profile.json"
            target.write_text(json.dumps({"provider": "vastai", "host": "root@gpu.example"}))
            profile.write_text(json.dumps(profile_data()))
            provider = Mock()
            provider.inspect.return_value = host_report()
            provider.plan.side_effect = VastProvider(Mock()).plan
            provider.deployment_inspect.return_value = facts()
            provider.deployment_blockers.side_effect = lambda value, config: remote_vast.blockers(value, config.as_dict())
            provider.deploy.return_value = {"model": "org/model", "endpoint": "http://127.0.0.1:8000/v1"}
            output = io.StringIO()
            with patch("inference_platform.cli.create_provider", return_value=provider), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value="deploy root@gpu.example small-model"), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = cli.main(["deploy", "--target", str(target), "--profile", str(profile)])
            self.assertEqual(code, 0)
            self.assertIn("cannot be independently verified", output.getvalue())
            provider.deploy.assert_called_once_with(parse_profile(profile_data()).as_dict())

    def test_prepare_vast_never_installs_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"provider": "vastai", "host": "root@gpu.example"}))
            provider = Mock()
            provider.inspect.return_value = host_report()
            provider.plan.side_effect = VastProvider(Mock()).plan
            with patch("inference_platform.cli.create_provider", return_value=provider), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["prepare", "--target", str(target)]), 0)
            provider.prepare.assert_not_called()

    def test_detached_process_readiness_stop_and_manual_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "vllm"
            executable.write_text("""#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
Path(os.environ["CUDA_OBSERVE_PATH"]).write_text(os.environ["CUDA_VISIBLE_DEVICES"])
model = sys.argv[2]
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.respond({"data": [{"id": model}]})
    def do_POST(self):
        self.respond({"choices": [{"text": "ready"}]})
    def respond(self, value):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
""")
            executable.chmod(0o700)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            profile = {**profile_data(), "port": port}
            supervisor = Path(remote_vast.__file__).with_name("remote_vast_supervisor.py")
            source = ("import runpy\n"
                      f"module = runpy.run_path({str(supervisor)!r})\n"
                      "module['main'].__globals__['proc_start'] = lambda pid: 'test-start'\n"
                      "module['main']()\n")
            tracked = []
            spawned = []
            real_spawn = os.posix_spawn

            def spawn(*args, **kwargs):
                pid = real_spawn(*args, **kwargs)
                spawned.append(pid)
                return pid

            def open_pidfd(pid):
                tracked.append(pid)
                return os.open(os.devnull, os.O_RDONLY)

            def send_pidfd(_fd, sig):
                os.kill(tracked[-1], sig)

            with patch.object(remote_vast, "ROOT", root / "state"), \
                    patch.object(remote_vast, "container_epoch", return_value=["boot", "1"]), \
                    patch.object(remote_vast, "supervisor_alive", return_value=True), \
                    patch.object(remote_vast, "require_ready"), \
                    patch.object(remote_vast.os, "posix_spawn", side_effect=spawn), \
                    patch.object(remote_vast.os, "pidfd_open", side_effect=open_pidfd, create=True), \
                    patch.object(remote_vast.signal, "pidfd_send_signal", side_effect=send_pidfd, create=True), \
                    patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"],
                                            "CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second",
                                            "CUDA_OBSERVE_PATH": str(root / "cuda.txt")}):
                try:
                    for create in (True, False):
                        remote_vast.start_process(profile, source, create=create)
                        self.assertEqual(remote_vast.wait_for_model(profile)["model"], profile["model"])
                        self.assertEqual((root / "cuda.txt").read_text(), "GPU-first")
                        self.assertTrue(remote_vast.status(profile)["running"])
                        stopped = remote_vast.manage("stop", profile)
                        self.assertEqual(stopped["state"], "exited")
                        os.waitpid(tracked[-1], 0)
                    self.assertEqual(remote_vast.manage("remove", profile)["state"], "not-found")
                finally:
                    for pid in spawned:
                        try:
                            completed, _ = os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            continue
                        if completed:
                            continue
                        os.kill(pid, signal.SIGTERM)
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            if os.waitpid(pid, os.WNOHANG)[0]:
                                break
                            time.sleep(0.1)


if __name__ == "__main__":
    unittest.main()
