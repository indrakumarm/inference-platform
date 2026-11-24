from dataclasses import dataclass
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

_TARGET_FIELDS = {"provider", "host", "port", "key", "known_hosts"}


@dataclass(frozen=True)
class Target:
    """How to reach one GPU host: the handoff from infrastructure tooling to this CLI."""

    host: str
    provider: str = "ssh"
    port: int = 22
    key: str | None = None
    known_hosts: str | None = None


def _resolve_path(value, base, field):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Target {field} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def parse_target(data, base=Path(".")):
    if not isinstance(data, dict):
        raise ValueError("Target must be a JSON object")
    extra = data.keys() - _TARGET_FIELDS
    if extra:
        raise ValueError("Invalid target (unknown fields: " + ", ".join(sorted(extra)) + ")")
    if not isinstance(data.get("host"), str) or not data["host"]:
        raise ValueError("Target host must be user@hostname or user@IPv4")
    provider = data.get("provider", "ssh")
    if not isinstance(provider, str) or not provider:
        raise ValueError("Target provider must be a provider name")
    port = data.get("port", 22)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Target port must be an integer between 1 and 65535")
    return Target(host=data["host"], provider=provider, port=port,
                  key=_resolve_path(data.get("key"), base, "key"),
                  known_hosts=_resolve_path(data.get("known_hosts"), base, "known_hosts"))


def load_target(path):
    """Load a target file; relative key/known_hosts paths resolve against the file's directory."""
    logger.info("Loading target from %s", path)
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid target JSON: {error}") from error
    return parse_target(data, path.resolve().parent)


class Provider:
    """Workload operations on one GPU host.

    The CLI drives every provider through this surface and owns reporting and
    confirmation. Each provider composes a host-readiness planner with a runtime
    (how vLLM is inspected, launched and managed on that kind of host).
    """

    name = ""

    def __init__(self, transport):
        self.transport = transport

    @property
    def target(self):
        return self.transport.target

    def inspect(self):
        raise NotImplementedError

    def plan(self, report):
        raise NotImplementedError

    def prepare(self, actions, root):
        raise NotImplementedError

    def deployment_inspect(self, profile):
        raise NotImplementedError

    def deployment_blockers(self, facts, profile):
        raise NotImplementedError

    def deploy(self, profile):
        raise NotImplementedError

    def lifecycle(self, operation, profile):
        raise NotImplementedError

    def deployment_list(self):
        raise NotImplementedError
