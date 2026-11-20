import contextlib
from email.message import Message
import io
import json
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from inference_platform import cli, gateway_check, remote_gateway
from inference_platform.gateway import render_gateway
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


class RemoteGatewayTests(unittest.TestCase):
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
