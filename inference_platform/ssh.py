import json
import logging
from pathlib import Path
import re
import shlex
import subprocess

from .logging_config import get_log_level


logger = logging.getLogger(__name__)


class SSHHost:
    """SSH transport for remote Python payloads.

    GPU-host workload operations live in ``providers``; this class only knows how to
    reach a host and run a command with a stdin payload. The small monitoring and
    gateway helpers below target hosts that are not managed by a provider.
    """

    def __init__(self, target, key=None, port=22, known_hosts=None):
        if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*", target):
            raise ValueError("Host must be user@hostname or user@IPv4 (no SSH options)")
        if not 1 <= port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        self.target = target
        self.log_level = get_log_level()
        self.command = ["ssh", "-T", "-o", "BatchMode=yes", "-o",
                        "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10",
                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                        "-p", str(port)]
        if known_hosts:
            path = Path(known_hosts).expanduser()
            if not path.is_file():
                raise ValueError(f"known_hosts file not found: {path}")
            self.command.extend(["-o", f"UserKnownHostsFile={path}"])
        if key:
            path = Path(key).expanduser()
            if not path.is_file():
                raise ValueError(f"SSH key file not found: {path}")
            self.command.extend(["-i", str(path)])
        self.command.append(target)

    def run(self, operation, remote_command, payload, timeout, **options):
        try:
            return subprocess.run(
                [*self.command, remote_command], input=payload,
                text=True, timeout=timeout, check=True, **options,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"SSH {operation} on {self.target} timed out after {timeout} seconds. "
                "The remote operation may still be running; check host status before retrying."
            ) from error
        except subprocess.CalledProcessError as error:
            detail = error.stderr or ""
            if isinstance(detail, bytes):
                detail = detail.decode(errors="replace")
            detail = detail.strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
            if error.returncode == 255:
                message = (
                    f"SSH connection/authentication failed for {self.target} during {operation}. "
                    "Check network access, host-key trust, and SSH credentials."
                )
            else:
                message = f"Remote {operation} failed on {self.target} (exit {error.returncode})."
            if detail:
                message += "\n" + detail
            raise RuntimeError(message) from error

    def update_monitoring_target(self, action, ip, port, name):
        source = Path(__file__).with_name("remote_monitoring.py").read_text()
        command = shlex.join(["python3", "-c", source, action])
        payload = json.dumps({"ip": ip, "port": port, "name": name})
        logger.info("Updating Prometheus target %s:%s on %s (%s)", ip, port, self.target, action)
        self.run(f"monitoring target {action}", command, payload, timeout=30, capture_output=True)

    def gateway_deploy(self, config):
        self._gateway_call("deploy", config)

    def gateway_bootstrap_tls(self, server_name, token):
        self._gateway_call("bootstrap-tls", json.dumps({"server_name": server_name, "token": token}))

    def gateway_inspect(self, port, model):
        result = self._gateway_call("inspect", json.dumps({"port": port, "model": model}))
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("Gateway host did not return a valid inspection response") from error
        if not isinstance(response, dict) or type(response.get("credentials_exist")) is not bool:
            raise ValueError("Gateway host returned an invalid inspection response")
        return response

    def gateway_certificate(self):
        return self._gateway_call("certificate", "").stdout

    def _gateway_call(self, action, payload):
        source = (Path(__file__).with_name("logging_config.py").read_text() + "\n" +
                  Path(__file__).with_name("remote_gateway.py").read_text())
        command = shlex.join([
            "sudo", "-n", "--", "env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
            "python3", "-c", source, action,
        ])
        logger.info("Starting remote gateway %s on %s", action, self.target)
        return self.run(f"gateway {action}", command, payload, timeout=120, stdout=subprocess.PIPE)
