import contextlib
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, Mock, patch

from inference_platform import cli, gateway_check, remote_gateway
from inference_platform.gateway import render_gateway
from inference_platform.platform_config import load_api_token, load_platform_config
from inference_platform.ssh import SSHHost


ROOT = Path(__file__).resolve().parents[1]


class GatewayTests(unittest.TestCase):
    def profile(self):
        return json.loads((ROOT / "gateway.json").read_text())

    def test_check_runs_without_ssh(self):
        profile = {**self.profile(), "tls": {"server_name": "inference.example.internal"}}
        output = io.StringIO()
        token = "ab" * 32
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gateway.json"
            path.write_text(json.dumps(profile))
            with patch.dict("os.environ", {"API_TOKEN": token}), \
                    patch("inference_platform.gateway.SSHHost") as ssh, \
                    patch("inference_platform.gateway.check_gateway", return_value=(0.5, 1.0)) as check, \
                    contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["gateway", "check", "--profile", str(path),
                                           "--model", "Qwen/Qwen3-8B"]), 0)
        check.assert_called_once_with("inference.example.internal", 8080, None, "Qwen/Qwen3-8B", token, 30)
        ssh.assert_not_called()
        self.assertIn("[DONE]", output.getvalue())
        self.assertNotIn(token, output.getvalue())

    def test_check_requires_401_then_generated_text_and_done(self):
        text = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n'
        role = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n'
        done = b"data: [DONE]\n"
        token = "ab" * 32
        for status, events, error in (
            (401, [role, text, done], None),
            (200, [], "expected HTTP 401"),
            (401, [text], r"without \[DONE\]"),
            (401, [role, done], "without generated text"),
        ):
            with self.subTest(status=status, error=error), \
                    patch("inference_platform.gateway_check.http.client.HTTPSConnection") as https, \
                    contextlib.redirect_stdout(io.StringIO()):
                unauthenticated, authenticated = https.return_value, MagicMock()
                https.side_effect = [unauthenticated, authenticated]
                unauthenticated.getresponse.return_value.__enter__.return_value.status = status
                response = authenticated.getresponse.return_value.__enter__.return_value
                response.status = 200
                response.headers = Message()
                response.headers["Content-Type"] = "text/event-stream"
                response.__iter__.return_value = iter(events)
                args = ("inference.example.internal", 443, None, "Qwen/Qwen3-8B", token, 5)
                if error:
                    with self.assertRaisesRegex(RuntimeError, error):
                        gateway_check.check_gateway(*args)
                else:
                    first, total = gateway_check.check_gateway(*args)
                    self.assertLessEqual(first, total)
                    self.assertNotIn("Authorization", unauthenticated.request.call_args.kwargs["headers"])
                    self.assertEqual(authenticated.request.call_args.kwargs["headers"]["Authorization"],
                                     f"Bearer {token}")

    def test_cli_renders_checked_in_config_without_ssh(self):
        output = io.StringIO()
        with patch("inference_platform.gateway.SSHHost") as ssh, contextlib.redirect_stdout(output):
            code = cli.main(["gateway", "render", "--profile", str(ROOT / "gateway.json")])
        self.assertEqual(code, 0)
        ssh.assert_not_called()
        self.assertEqual(output.getvalue(), (ROOT / "gateway/nginx/inference.conf").read_text())

    def test_multiple_backends_and_custom_limits_render(self):
        profile = self.profile()
        profile.update(
            backends=[{"host": "gpu.example", "port": 8101}, {"host": "::1", "port": 8102}],
            max_in_flight=2,
            connect_timeout_seconds=8,
            upstream_read_timeout_seconds=90,
        )
        output = render_gateway(profile)
        for directive in (
            "server gpu.example:8101;", "server [::1]:8102;", "limit_conn inference_inflight 2;",
            "proxy_connect_timeout 8s;", "proxy_read_timeout 90s;",
        ):
            with self.subTest(directive=directive):
                self.assertIn(directive, output)

    def test_invalid_profiles_are_rejected(self):
        for overrides, message in (
            ({"listen": {"host": "0.0.0.0", "port": 8080}}, "listen.host"),
            ({"backends": []}, "non-empty list"),
            ({"backends": [{"host": "bad;name", "port": 8000}]}, "hostname"),
            ({"backends": [{"host": "unix", "port": 8000}]}, "hostname"),
            ({"backends": [{"host": "127.0.0.1", "port": 8080}]}, "gateway listener"),
            ({"max_in_flight": True}, "max_in_flight"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                render_gateway({**self.profile(), **overrides})

    def test_https_requires_authentication_and_does_not_forward_credentials(self):
        output = render_gateway({
            **self.profile(),
            "listen": {"host": "10.0.1.10", "port": 443},
            "tls": {"server_name": "inference.example.internal"},
        })
        self.assertIn("listen 10.0.1.10:443 ssl;", output)
        self.assertIn("server_name inference.example.internal;", output)
        self.assertIn("map $http_authorization $inference_bearer_allowed {\n    default 0;", output)
        self.assertIn("include /etc/nginx/inference-platform/bearer-token.map;", output)
        self.assertIn("return 401;", output)
        self.assertIn('proxy_set_header Authorization "";', output)
        self.assertIn("proxy_buffering off;", output)

    def test_https_is_required_for_port_443(self):
        profile = self.profile()
        profile["listen"]["port"] = 443
        with self.assertRaisesRegex(ValueError, r"listen\.port"):
            render_gateway(profile)

    def test_gateway_access_log_records_timings_without_request_content(self):
        for tls in ({}, {"tls": {"server_name": "inference.example.internal"}}):
            with self.subTest(tls=bool(tls)):
                output = render_gateway({**self.profile(), **tls})
                self.assertIn("log_format inference_timing ", output)
                log_format = output.split("log_format inference_timing ", 1)[1].split(";", 1)[0]
                for field in (
                    "$request_id", "$status", "$upstream_status", "$request_time",
                    "$upstream_connect_time", "$upstream_header_time", "$upstream_response_time",
                ):
                    self.assertIn(field, log_format)
                for field in ("$http_authorization", "$request_body", "$request_uri", "$args"):
                    self.assertNotIn(field, log_format)
                self.assertIn("access_log /var/log/nginx/inference_access.log inference_timing;", output)

    def test_tls_rejects_wildcard_public_or_invalid_listeners_and_unsafe_names(self):
        for host, name in (
            ("0.0.0.0", "inference.example.internal"),
            ("8.8.8.8", "inference.example.internal"),
            ("10.0.1.10", "example.internal;"),
        ):
            with self.subTest(host=host, name=name), self.assertRaises(ValueError):
                render_gateway({
                    **self.profile(), "listen": {"host": host, "port": 8443},
                    "tls": {"server_name": name},
                })
        with self.assertRaisesRegex(ValueError, "listen.host"):
            render_gateway({**self.profile(), "listen": {"host": "10.0.1.10", "port": 8443}})

    def test_deploy_sends_render_only_after_confirmation(self):
        argv = ["gateway", "deploy", "--host", "ubuntu@gpu.example",
                "--profile", str(ROOT / "gateway.json")]
        for terminal, answer, expected in (
            (False, None, 2), (True, "no", 0), (True, "", 0),
            (True, "Y", 0), (True, " yes ", 0),
        ):
            with self.subTest(terminal=terminal, answer=answer), \
                    patch("inference_platform.gateway.SSHHost") as ssh, \
                    patch("sys.stdin.isatty", return_value=terminal), \
                    patch("builtins.input", return_value=answer) as prompt, \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(argv), expected)
                if terminal:
                    prompt.assert_called_once_with(
                        "Deploy config and reload NGINX on ubuntu@gpu.example? [y/N]: ")
                if answer in ("Y", " yes "):
                    ssh.return_value.gateway_deploy.assert_called_once_with(render_gateway(self.profile()))
                else:
                    ssh.return_value.gateway_deploy.assert_not_called()
                if not terminal:
                    prompt.assert_not_called()

    def test_transport_uses_checked_ssh_and_sends_config_on_stdin(self):
        config = render_gateway(self.profile())
        with patch.dict("os.environ", {"INFERENCE_PLATFORM_LOG_LEVEL": "INFO"}), \
                patch("inference_platform.ssh.subprocess.run") as run:
            SSHHost("ubuntu@gpu.example").gateway_deploy(config)
        command = run.call_args.args[0]
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertTrue(command[-1].startswith(
            "sudo -n -- env INFERENCE_PLATFORM_LOG_LEVEL=INFO python3 -c "))
        self.assertNotIn("stderr", run.call_args.kwargs)
        self.assertFalse(run.call_args.kwargs.get("capture_output", False))
        self.assertEqual(run.call_args.kwargs["input"], config)

    def test_gateway_inspection_uses_strict_ssh_and_validates_response(self):
        with patch("inference_platform.ssh.subprocess.run") as run:
            run.return_value.stdout = '{"credentials_exist": true}'
            host = SSHHost("ubuntu@gpu.example")
            self.assertTrue(host.gateway_inspect(8000, "org/model")["credentials_exist"])
            self.assertIn("StrictHostKeyChecking=yes", run.call_args.args[0])
            self.assertEqual(json.loads(run.call_args.kwargs["input"]),
                             {"port": 8000, "model": "org/model"})
            run.return_value.stdout = "{}"
            with self.assertRaisesRegex(ValueError, "invalid inspection response"):
                host.gateway_inspect(8000, "org/model")

    def test_bootstrap_prompts_for_token_and_sends_it_only_on_ssh_stdin(self):
        token = "ab" * 32
        profile = {**self.profile(), "tls": {"server_name": "inference.example.internal"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gateway.json"
            path.write_text(json.dumps(profile))
            output = io.StringIO()
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value="yes"), \
                    patch("inference_platform.gateway.getpass.getpass", return_value=token), \
                    patch("inference_platform.ssh.subprocess.run") as run, \
                    contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["gateway", "bootstrap-tls", "--host",
                                           "ubuntu@gpu.example", "--profile", str(path)]), 0)
            run.assert_called_once()
            self.assertNotIn(token, run.call_args.args[0][-1])
            self.assertNotIn(token, output.getvalue())
            self.assertEqual(json.loads(run.call_args.kwargs["input"]), {
                "server_name": "inference.example.internal", "token": token,
            })

    def test_guided_http_setup_uses_target_backend_and_verifies_client_stream(self):
        output = io.StringIO()
        host = Mock()
        host.gateway_inspect.return_value = {"credentials_exist": False}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example", "port": 2222}))
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["http", "", "yes", "yes"]), \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host) as ssh, \
                    patch("inference_platform.gateway_setup.check_http_gateway",
                          return_value=(0.2, 0.5)) as check, \
                    contextlib.redirect_stdout(output):
                result = cli.main(["gateway", "setup", "--target", str(target),
                                   "--deployment-profile", str(ROOT / "deployment.json")])
        self.assertEqual(result, 0)
        ssh.assert_called_once_with("ubuntu@gpu.example", None, 2222, None)
        host.gateway_inspect.assert_called_once_with(8000, "Qwen/Qwen3-8B")
        host.gateway_deploy.assert_called_once()
        self.assertIn("server 127.0.0.1:8000;", host.gateway_deploy.call_args.args[0])
        check.assert_called_once_with("127.0.0.1", 8080, "Qwen/Qwen3-8B", 30)
        self.assertIn("SSH tunnel", output.getvalue())

    def test_guided_https_creates_credentials_after_confirmation_and_verifies_without_leaking_token(self):
        token = "ab" * 32
        host = Mock()
        host.gateway_inspect.return_value = {"credentials_exist": False}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example"}))
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["https", "10.0.1.10", "443",
                          "inference.example.internal", "yes", "yes", "yes", "/tmp/public.pem"]), \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host), \
                    patch("inference_platform.gateway_setup.getpass.getpass", return_value=token), \
                    patch("inference_platform.gateway_setup.check_gateway",
                          return_value=(0.2, 0.5)) as check, \
                    contextlib.redirect_stdout(output):
                result = cli.main(["gateway", "setup", "--target", str(target),
                                   "--deployment-profile", str(ROOT / "deployment.json")])
        self.assertEqual(result, 0)
        host.gateway_bootstrap_tls.assert_called_once_with("inference.example.internal", token)
        self.assertLess(host.mock_calls.index(unittest.mock.call.gateway_bootstrap_tls(
            "inference.example.internal", token)), host.mock_calls.index(
                unittest.mock.call.gateway_deploy(host.gateway_deploy.call_args.args[0])))
        check.assert_called_once_with("inference.example.internal", 443, Path("/tmp/public.pem"),
                                      "Qwen/Qwen3-8B", token, 30)
        self.assertNotIn(token, output.getvalue())

    def test_guided_setup_cancel_preflight_failure_and_partial_failure_are_explicit(self):
        for scenario in ("cancel", "preflight", "deploy"):
            host = Mock()
            host.gateway_inspect.return_value = {"credentials_exist": False}
            if scenario == "preflight":
                host.gateway_inspect.side_effect = RuntimeError("model missing")
            if scenario == "deploy":
                host.gateway_deploy.side_effect = RuntimeError("reload failed")
            answers = ["https", "", "", "inference.example.internal"]
            if scenario != "preflight":
                answers += ["yes", "no" if scenario == "cancel" else "yes"]
            output = io.StringIO()
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "target.json"
                target.write_text(json.dumps({"host": "ubuntu@gpu.example"}))
                with patch("sys.stdin.isatty", return_value=True), \
                        patch("builtins.input", side_effect=answers), \
                        patch("inference_platform.gateway_setup.SSHHost", return_value=host), \
                        patch("inference_platform.gateway_setup.getpass.getpass", return_value="ab" * 32), \
                        contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    result = cli.main(["gateway", "setup", "--target", str(target),
                                       "--deployment-profile", str(ROOT / "deployment.json")])
            self.assertEqual(result, 0 if scenario == "cancel" else 1)
            if scenario != "deploy":
                host.gateway_bootstrap_tls.assert_not_called()
                host.gateway_deploy.assert_not_called()
            else:
                self.assertIn("credentials were created but deployment did not complete", output.getvalue())

    def test_verify_uses_existing_credentials_without_ssh_or_gateway_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example"}))
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["https", "443", "inference.example.internal", ""]), \
                    patch("inference_platform.gateway_setup.getpass.getpass", return_value="ab" * 32), \
                    patch("inference_platform.gateway_setup.SSHHost") as ssh, \
                    patch("inference_platform.gateway_setup.check_gateway", return_value=(0.1, 0.2)) as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["gateway", "verify", "--target", str(target),
                                           "--deployment-profile", str(ROOT / "deployment.json")]), 0)
        ssh.assert_not_called()
        check.assert_called_once_with("inference.example.internal", 443, None,
                                      "Qwen/Qwen3-8B", "ab" * 32, 30)

    def test_existing_https_credentials_are_reused_without_overwrite(self):
        host = Mock()
        host.gateway_inspect.return_value = {"credentials_exist": True}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example"}))
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["https", "", "", "inference.example.internal",
                                                         "yes", "yes", "no"]), \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["gateway", "setup", "--target", str(target),
                                           "--deployment-profile", str(ROOT / "deployment.json")]), 0)
        host.gateway_bootstrap_tls.assert_not_called()
        host.gateway_deploy.assert_called_once()
        self.assertIn("client verification skipped", output.getvalue())

    def test_setup_refuses_noninteractive_execution_before_remote_access(self):
        with patch("sys.stdin.isatty", return_value=False), \
                patch("inference_platform.gateway_setup.SSHHost") as ssh, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["gateway", "setup", "--target", "missing.json",
                                       "--deployment-profile", "missing.json"]), 2)
        ssh.assert_not_called()

    def test_verification_failure_reports_gateway_already_deployed(self):
        host = Mock()
        host.gateway_inspect.return_value = {"credentials_exist": False}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example"}))
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["http", "", "yes", "yes"]), \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host), \
                    patch("inference_platform.gateway_setup.check_http_gateway",
                          side_effect=RuntimeError("stream failed")), \
                    contextlib.redirect_stderr(output), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["gateway", "setup", "--target", str(target),
                                           "--deployment-profile", str(ROOT / "deployment.json")]), 1)
        host.gateway_deploy.assert_called_once()
        self.assertIn("NGINX was reloaded, but client verification failed", output.getvalue())

    def test_platform_config_setup_reuses_files_and_env_without_repeating_settings(self):
        token = "ab" * 32
        host = Mock()
        host.gateway_inspect.return_value = {"credentials_exist": False}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target.json").write_text(json.dumps({"host": "ubuntu@gpu.example"}))
            (root / "deployment.json").write_text((ROOT / "deployment.json").read_text())
            config = root / "platform.json"
            config.write_text(json.dumps({
                "target": "target.json", "deployment": "deployment.json",
                "gateway": {"mode": "https", "listen": {"host": "10.0.1.10", "port": 443},
                            "tls": {"server_name": "inference.example.internal"},
                            "ca_file": "public.pem"}}))
            (root / ".env").write_text("API_TOKEN=" + token + "\n")
            with patch.dict(os.environ, {}, clear=True), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", side_effect=["yes", "yes", "no"]) as prompt, \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host), \
                    patch("inference_platform.gateway_setup.getpass.getpass") as secret_prompt, \
                    contextlib.redirect_stdout(output):
                result = cli.main(["gateway", "setup", "--config", str(config)])
        self.assertEqual(result, 0)
        self.assertEqual(prompt.call_count, 3)
        secret_prompt.assert_not_called()
        host.gateway_bootstrap_tls.assert_called_once_with("inference.example.internal", token)
        self.assertIn("--config", output.getvalue())
        self.assertNotIn(token, output.getvalue())

    def test_platform_verify_is_noninteractive_and_env_overrides_dotenv(self):
        file_token, env_token = "ab" * 32, "cd" * 32
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "platform.json"
            config.write_text(json.dumps({
                "target": {"host": "ubuntu@gpu.example"},
                "deployment": json.loads((ROOT / "deployment.json").read_text()),
                "gateway": {"mode": "https", "listen": {"host": "10.0.1.10", "port": 443},
                            "tls": {"server_name": "inference.example.internal"},
                            "ca_file": "public.pem"}}))
            (root / ".env").write_text(f"API_TOKEN='{file_token}'\n")
            with patch.dict(os.environ, {"API_TOKEN": env_token}, clear=True), \
                    patch("sys.stdin.isatty", return_value=False), \
                    patch("builtins.input") as prompt, \
                    patch("inference_platform.gateway_setup.check_gateway",
                          return_value=(0.1, 0.2)) as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 0)
            check.assert_called_once_with("inference.example.internal", 443,
                                          (root / "public.pem").resolve(),
                                          "Qwen/Qwen3-8B", env_token, 30)
            prompt.assert_not_called()

    def test_platform_config_missing_field_prompts_or_fails_explicitly_without_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "platform.json"
            config.write_text(json.dumps({"target": {"host": "ubuntu@gpu.example"},
                                          "deployment": json.loads((ROOT / "deployment.json").read_text()),
                                          "gateway": {"mode": "http"}}))
            output = io.StringIO()
            with patch("sys.stdin.isatty", return_value=False), \
                    patch("inference_platform.gateway_setup.check_http_gateway") as check, \
                    contextlib.redirect_stderr(output):
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 1)
            check.assert_not_called()
            with patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value="8080") as prompt, \
                    patch("inference_platform.gateway_setup.check_http_gateway",
                          return_value=(0.1, 0.2)) as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 0)
            prompt.assert_called_once()
            check.assert_called_once_with("127.0.0.1", 8080, "Qwen/Qwen3-8B", 30)

    def test_invalid_dotenv_token_fails_before_network_or_printing_secret(self):
        secret = "not-a-valid-token"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "platform.json"
            config.write_text(json.dumps({
                "target": {"host": "ubuntu@gpu.example"},
                "deployment": json.loads((ROOT / "deployment.json").read_text()),
                "gateway": {"mode": "https", "listen": {"host": "127.0.0.1", "port": 8443},
                            "tls": {"server_name": "inference.example.internal"},
                            "ca_file": "public.pem", "fetch_lab_certificate": True}}))
            (root / ".env").write_text(f"API_TOKEN={secret}\n")
            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), \
                    patch("sys.stdin.isatty", return_value=False), \
                    patch("inference_platform.gateway_setup.SSHHost") as ssh, \
                    patch("inference_platform.gateway_setup.check_gateway") as check, \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 1)
            ssh.assert_not_called()
            check.assert_not_called()
            self.assertNotIn(secret, output.getvalue())

    def test_target_config_cannot_be_combined_with_individual_paths(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["gateway", "verify", "--config", "config.json", "--target", "target.json"])

    def test_platform_config_rejects_unknown_fields_and_relative_paths_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "platform.json"
            data = {"target": {"host": "ubuntu@gpu.example", "key": "id.pem"},
                    "deployment": json.loads((ROOT / "deployment.json").read_text()),
                    "gateway": {"mode": "https", "ca_file": "cert.pem"}}
            config.write_text(json.dumps(data))
            parsed = load_platform_config(config)
            self.assertEqual(parsed.target.key, str((root / "id.pem").resolve()))
            self.assertEqual(parsed.gateway["ca_file"], (root / "cert.pem").resolve())
            data["gateway"]["token"] = "not-allowed"
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "gateway configuration"):
                load_platform_config(config)

    def test_dotenv_does_not_execute_shell_and_never_echoes_invalid_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("OTHER=value\nAPI_TOKEN=$(touch should-not-exist)\n")
            self.assertEqual(load_api_token(env), "$(touch should-not-exist)")
            env.write_text("API_TOKEN=" + "ab" * 32 + "\nAPI_TOKEN=" + "cd" * 32)
            with self.assertRaisesRegex(ValueError, "Duplicate API_TOKEN"):
                load_api_token(env)

    @unittest.skipUnless(shutil.which("openssl"), "requires OpenSSL")
    def test_config_fetches_public_certificate_once_via_strict_ssh(self):
        import ssl

        token = "ab" * 32
        host = Mock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(remote_gateway, "TLS_DIR", root / "credentials"):
                remote_gateway._install_lab_credentials(("inference.example.internal", token))
                certificate = remote_gateway.public_certificate()
            host.gateway_certificate.return_value = certificate
            target = root / "target.json"
            target.write_text(json.dumps({"host": "ubuntu@gpu.example", "port": 2222}))
            config = root / "platform.json"
            config.write_text(json.dumps({
                "target": "target.json", "deployment": str(ROOT / "deployment.json"),
                "gateway": {"mode": "https", "listen": {"host": "10.0.1.10", "port": 443},
                            "tls": {"server_name": "inference.example.internal"},
                            "ca_file": "public.pem", "fetch_lab_certificate": True}}))
            with patch.dict(os.environ, {"API_TOKEN": token}, clear=True), \
                    patch("sys.stdin.isatty", return_value=False), \
                    patch("inference_platform.gateway_setup.SSHHost", return_value=host) as ssh, \
                    patch("inference_platform.gateway_setup.check_gateway", return_value=(0.1, 0.2)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 0)
                self.assertEqual(cli.main(["gateway", "verify", "--config", str(config)]), 0)
            ssh.assert_called_once_with("ubuntu@gpu.example", None, 2222, None)
            host.gateway_certificate.assert_called_once()
            cert = root / "public.pem"
            self.assertEqual(cert.read_text(), certificate)
            self.assertEqual(stat.S_IMODE(cert.stat().st_mode), 0o600)
            ssl.create_default_context(cafile=str(cert))

    def test_http_verification_requires_streamed_text_and_done_without_authentication(self):
        with patch("inference_platform.gateway_check.http.client.HTTPConnection") as http:
            response = http.return_value.getresponse.return_value.__enter__.return_value
            response.status = 200
            response.headers = Message()
            response.headers["Content-Type"] = "text/event-stream"
            response.__iter__.return_value = iter([
                b'data: {"choices":[{"delta":{"content":"OK"}}]}\n', b"data: [DONE]\n"])
            gateway_check.check_http_gateway("127.0.0.1", 8080, "org/model", 5)
        self.assertNotIn("Authorization", http.return_value.request.call_args.kwargs["headers"])

    def test_http_verification_against_local_streaming_server(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, self.headers.get("Authorization"),
                                 json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                                 b'data: [DONE]\n\n')

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            first, total = gateway_check.check_http_gateway("127.0.0.1", server.server_port, "org/model", 5)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertLessEqual(first, total)
        self.assertEqual(requests[0][0:2], ("/v1/chat/completions", None))
        self.assertEqual(requests[0][2]["model"], "org/model")
        self.assertTrue(requests[0][2]["stream"])


class RemoteGatewayTests(unittest.TestCase):
    def test_public_certificate_reader_only_accepts_regular_certificate_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / "credentials"
            credentials.mkdir()
            certificate = credentials / "fullchain.pem"
            certificate.write_text("public certificate")
            with patch.object(remote_gateway, "TLS_DIR", credentials):
                self.assertEqual(remote_gateway.public_certificate(), "public certificate")
                certificate.unlink()
                certificate.symlink_to(root / "other")
                with self.assertRaisesRegex(RuntimeError, "regular file"):
                    remote_gateway.public_certificate()

    def test_read_only_inspection_checks_nginx_and_model_identity(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(remote_gateway, "SITE", Path(directory) / "site.conf"), \
                patch.object(remote_gateway, "TLS_DIR", Path(directory) / "credentials"), \
                patch.object(remote_gateway, "run_checked") as run, \
                patch.object(remote_gateway.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(
                b'{"data":[{"id":"org/model"}]}')
            self.assertEqual(remote_gateway.inspect_gateway({"port": 8000, "model": "org/model"}),
                             {"credentials_exist": False})
            run.assert_called_once_with(["systemctl", "is-active", "--quiet", "nginx"])
            urlopen.return_value.__enter__.return_value = io.BytesIO(
                b'{"data":[{"id":"org/model"}]}')
            with self.assertRaisesRegex(RuntimeError, "expected model"):
                remote_gateway.inspect_gateway({"port": 8000, "model": "different"})
            with patch.object(remote_gateway, "SITE", Path(directory) / "missing" / "site.conf"):
                with self.assertRaisesRegex(RuntimeError, "install and configure NGINX"):
                    remote_gateway.inspect_gateway({"port": 8000, "model": "org/model"})

    @unittest.skipUnless(shutil.which("openssl"), "requires OpenSSL")
    def test_bootstrap_installs_trustable_matching_certificate_and_protects_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / "inference-platform"
            hostname = "inference.example.internal"
            token = "ab" * 32
            with patch.object(remote_gateway, "SITE", root / "site.conf"), \
                    patch.object(remote_gateway, "TLS_DIR", credentials):
                remote_gateway.bootstrap_tls({"server_name": hostname, "token": token})
                with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                    remote_gateway.bootstrap_tls({"server_name": hostname, "token": "cd" * 32})
            cert = credentials / "fullchain.pem"
            key = credentials / "privkey.pem"
            token_map = credentials / "bearer-token.map"
            subprocess.run(["openssl", "verify", "-CAfile", str(cert), "-verify_hostname",
                            hostname, str(cert)], check=True, capture_output=True)
            self.assertEqual(
                subprocess.check_output(["openssl", "x509", "-in", str(cert), "-pubkey", "-noout"]),
                subprocess.check_output(["openssl", "pkey", "-in", str(key), "-pubout"]),
            )
            for path in (key, token_map):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(token_map.read_text(), f'"~^Bearer[ ]{token}$" 1;\n')

    def test_deploy_validates_then_reloads_and_rolls_back_on_failure(self):
        for failure, existing in (
            (None, False), ("validation", True), ("conflict", True), ("not included", False),
            ("reload", True),
        ):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                site = Path(directory) / "inference-platform.conf"
                old = remote_gateway.MANAGED + b"old"
                new = remote_gateway.MANAGED + b"new"
                if existing:
                    site.write_bytes(old)
                commands = []

                def run(command):
                    commands.append(command)
                    if command == ["nginx", "-T"]:
                        self.assertEqual(site.read_bytes(), new)
                        if failure == "validation":
                            raise RuntimeError("invalid config")
                        stderr = "conflicting server name" if failure == "conflict" else ""
                        stdout = "" if failure == "not included" else f"# configuration file {site}:\n"
                        return subprocess.CompletedProcess(command, 0, stdout, stderr)
                    if command == ["systemctl", "reload", "nginx"] and failure == "reload":
                        raise RuntimeError("reload failed")

                with patch.object(remote_gateway, "SITE", site), \
                        patch.object(remote_gateway, "run_checked", side_effect=run), \
                        contextlib.redirect_stderr(io.StringIO()):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            remote_gateway.deploy_gateway(b"new")
                    else:
                        with self.assertLogs("inference_platform.remote_gateway", "INFO") as logs:
                            remote_gateway.deploy_gateway(b"new")
                        self.assertIn("Checking that the NGINX service is active", logs.records[0].message)
                        self.assertIn("Validation passed; requesting NGINX reload", "\n".join(logs.output))

                expected = [["systemctl", "is-active", "--quiet", "nginx"], ["nginx", "-T"]]
                if failure in (None, "reload"):
                    expected.append(["systemctl", "reload", "nginx"])
                self.assertEqual(commands, expected)
                if failure is None or existing:
                    self.assertEqual(site.read_bytes(), old if failure else new)
                else:
                    self.assertFalse(site.exists())

    def test_unmanaged_site_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "inference-platform.conf"
            site.write_bytes(b"manual site")
            with patch.object(remote_gateway, "SITE", site), \
                    patch.object(remote_gateway, "run_checked", new_callable=Mock) as run:
                with self.assertRaisesRegex(RuntimeError, "unmanaged site"):
                    remote_gateway.deploy_gateway(b"new")
            run.assert_not_called()
            self.assertEqual(site.read_bytes(), b"manual site")
