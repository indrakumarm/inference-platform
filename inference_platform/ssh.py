import json
import logging
from pathlib import Path
import re
import shlex
import subprocess

from .logging_config import get_log_level


logger = logging.getLogger(__name__)


class SSHHost:
    """SSH transport; remote operations are supplied by the host adapter."""

    def __init__(self, target, key=None, port=22):
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
        if key:
            path = Path(key).expanduser()
            if not path.is_file():
                raise ValueError(f"SSH key file not found: {path}")
            self.command.extend(["-i", str(path)])
        self.command.append(target)
        logging_source = Path(__file__).with_name("logging_config.py").read_text()
        self.source = (logging_source + "\n" +
                       Path(__file__).with_name("remote.py").read_text())
        self.deployment_source = (logging_source + "\n" +
                                 Path(__file__).with_name("remote_deploy.py").read_text())

    def _run_ssh(self, operation, remote_command, payload, timeout, **options):
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

    def inspect(self):
        logger.info("Inspecting remote host %s", self.target)
        command = shlex.join(["env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
                              "python3", "-", "inspect"])
        logger.debug("Starting read-only host inspection over SSH")
        result = self._run_ssh("inspect", command, self.source, timeout=240,
                               capture_output=True)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("Host did not return JSON; check Python 3 and SSH login output") from error
        if not isinstance(report, dict) or report.get("schema") != 1:
            raise ValueError("Unsupported host report format")
        logger.debug("Received remote host inspection report")
        return report

    def prepare(self, actions, root):
        logger.info("Preparing remote host %s with actions: %s", self.target, actions)
        command = ([] if root else ["sudo", "-n", "--"]) + [
            "env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
            "python3", "-", "prepare", *actions]
        self._run_ssh("prepare", shlex.join(command), self.source, timeout=2400)
        logger.info("Remote host preparation command completed")

    def _deployment_call(self, operation, profile, timeout):
        logger.info("Starting remote deployment %s on %s", operation, self.target)
        remote_command = shlex.join([
            "env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
            "python3", "-c", self.deployment_source, operation])
        output_options = ({"stdout": subprocess.PIPE} if operation == "deploy"
                         else {"capture_output": True})
        result = self._run_ssh(operation, remote_command, json.dumps(profile),
                               timeout=timeout, **output_options)
        try:
            response = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError("Host did not return a valid deployment response") from error
        logger.debug("Received remote deployment %s response", operation)
        return response

    def deployment_inspect(self, profile):
        return self._deployment_call("inspect", profile, timeout=240)

    def deployment_list(self):
        return self._deployment_call("list", {}, timeout=240)

    def deploy(self, profile):
        return self._deployment_call("deploy", profile, timeout=7200)

    def lifecycle(self, operation, profile):
        return self._deployment_call(operation, profile, timeout=300)
