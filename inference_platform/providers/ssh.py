import json
import logging
from pathlib import Path
import shlex
import subprocess

from ..deployment import deployment_blockers
from ..ubuntu import plan
from .base import Provider


logger = logging.getLogger(__name__)

_PACKAGE = Path(__file__).resolve().parent.parent


class SSHProvider(Provider):
    """A dedicated Ubuntu host reached over SSH, running vLLM in rootful Docker."""

    name = "ssh"

    def __init__(self, transport):
        super().__init__(transport)
        logging_source = (_PACKAGE / "logging_config.py").read_text()
        self.source = logging_source + "\n" + (_PACKAGE / "remote.py").read_text()
        self.deployment_source = logging_source + "\n" + (_PACKAGE / "remote_deploy.py").read_text()

    @property
    def log_level(self):
        return self.transport.log_level

    def inspect(self):
        logger.info("Inspecting remote host %s", self.target)
        command = shlex.join(["env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
                              "python3", "-", "inspect"])
        logger.debug("Starting read-only host inspection over SSH")
        result = self.transport.run("inspect", command, self.source, timeout=240,
                                    capture_output=True)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("Host did not return JSON; check Python 3 and SSH login output") from error
        if not isinstance(report, dict) or report.get("schema") != 1:
            raise ValueError("Unsupported host report format")
        logger.debug("Received remote host inspection report")
        return report

    def plan(self, report):
        return plan(report)

    def prepare(self, actions, root):
        logger.info("Preparing remote host %s with actions: %s", self.target, actions)
        command = ([] if root else ["sudo", "-n", "--"]) + [
            "env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
            "python3", "-", "prepare", *actions]
        self.transport.run("prepare", shlex.join(command), self.source, timeout=2400)
        logger.info("Remote host preparation command completed")

    def _deployment_call(self, operation, profile, timeout):
        logger.info("Starting remote deployment %s on %s", operation, self.target)
        remote_command = shlex.join([
            "env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.log_level}",
            "python3", "-c", self.deployment_source, operation])
        output_options = ({"stdout": subprocess.PIPE} if operation == "deploy"
                          else {"capture_output": True})
        result = self.transport.run(operation, remote_command, json.dumps(profile),
                                    timeout=timeout, **output_options)
        try:
            response = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError("Host did not return a valid deployment response") from error
        logger.debug("Received remote deployment %s response", operation)
        return response

    def deployment_inspect(self, profile):
        return self._deployment_call("inspect", profile, timeout=240)

    def deployment_blockers(self, facts, profile):
        return deployment_blockers(facts, profile)

    def deployment_list(self):
        return self._deployment_call("list", {}, timeout=240)

    def deploy(self, profile):
        return self._deployment_call("deploy", profile, timeout=7200)

    def lifecycle(self, operation, profile):
        return self._deployment_call(operation, profile, timeout=300)
