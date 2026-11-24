import json
from pathlib import Path
import shlex
import subprocess

from ..remote_vast import blockers
from ..ubuntu import Plan
from .base import Provider


_PACKAGE = Path(__file__).resolve().parent.parent


class VastProvider(Provider):
    """vLLM processes inside an existing Vast container."""

    name = "vastai"

    def __init__(self, transport):
        super().__init__(transport)
        common = (_PACKAGE / "remote_deploy.py").read_text()
        logging_source = (_PACKAGE / "logging_config.py").read_text()
        self.source = (
            "import sys, types\n"
            + "for name, source in " + repr((("logging_config", logging_source),
                                               ("remote_deploy", common))) + ":\n"
            + "    module = types.ModuleType(name)\n"
            + "    sys.modules[name] = module\n"
            + "    exec(source, module.__dict__)\n"
            + (_PACKAGE / "remote_vast.py").read_text()
        )
        self.supervisor = (_PACKAGE / "remote_vast_supervisor.py").read_text()

    def _call(self, operation, profile=None, timeout=240):
        command = shlex.join(["env", f"INFERENCE_PLATFORM_LOG_LEVEL={self.transport.log_level}",
                              "python3", "-c", self.source, operation])
        payload = {"profile": profile or {}}
        if operation in ("deploy", "start"):
            payload["supervisor"] = self.supervisor
        options = {"stdout": subprocess.PIPE} if operation in ("deploy", "start") else {"capture_output": True}
        result = self.transport.run(operation, command, json.dumps(payload), timeout=timeout, **options)
        try:
            response = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError("Vast container did not return a valid workload response") from error
        if not isinstance(response, list if operation == "list" else dict):
            raise ValueError("Vast container returned an unexpected workload response")
        return response

    def inspect(self):
        report = self._call("host-inspect")
        if not isinstance(report, dict) or report.get("schema") != 1 or report.get("runtime") != "vast-container":
            raise ValueError("Unsupported Vast host report format")
        return report

    def plan(self, report):
        checks = report["checks"]
        blocked = []
        for name in ("gpu", "gpu_processes", "torch", "vllm", "cache"):
            result = checks[name]
            if result["code"] != 0:
                blocked.append(f"{name} check failed: {result['stderr'] or result['stdout']}")
        if checks["gpu"]["code"] == 0 and not checks["gpu"]["stdout"]:
            blocked.append("No NVIDIA GPUs are visible.")
        if checks["gpu_processes"]["code"] == 0 and checks["gpu_processes"]["stdout"]:
            blocked.append("GPU compute processes are active; deployment requires an idle container.")
        if checks["torch"]["code"] == 0:
            try:
                torch = json.loads(checks["torch"]["stdout"])
            except (TypeError, ValueError):
                blocked.append("PyTorch GPU probe returned invalid JSON.")
            else:
                if (not isinstance(torch, dict) or torch.get("available") is not True
                        or type(torch.get("count")) is not int or torch["count"] < 1):
                    blocked.append("PyTorch cannot access CUDA GPUs.")
        return Plan(blocked, [])

    def prepare(self, actions, root):
        if actions:
            raise ValueError("Vast container preparation does not install or change packages")

    def deployment_inspect(self, profile):
        return self._call("inspect", profile)

    def deployment_blockers(self, facts, profile):
        return blockers(facts, profile.as_dict())

    def deploy(self, profile):
        return self._call("deploy", profile, timeout=7200)

    def lifecycle(self, operation, profile):
        return self._call(operation, profile, timeout=7200 if operation == "start" else 300)

    def deployment_list(self):
        return self._call("list")
