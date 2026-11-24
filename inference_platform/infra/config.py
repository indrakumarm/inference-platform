from dataclasses import dataclass
import json
from pathlib import Path
import re

from ..providers.base import _resolve_path


@dataclass(frozen=True)
class AWSConfig:
    name: str
    region: str
    instance_type: str
    subnet_id: str
    security_group_ids: list[str]
    key_name: str
    root_gb: int
    ami_id: str | None = None
    ami_ssm: str | None = None
    profile: str | None = None
    project: str = "inference-platform"
    market: str = "ondemand"
    instance_profile: str | None = None
    private_ip: bool = False
    ssh_user: str = "ubuntu"
    key: str | None = None
    known_hosts: str | None = None
    state_dir: str = "~/.local/state/inference-platform/infra"


def load_config(path: Path) -> AWSConfig:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid infra JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Infra config must be a JSON object")
    required = {"name", "region", "instance_type", "subnet_id",
                "security_group_ids", "key_name", "root_gb"}
    missing = required - data.keys()
    extra = data.keys() - AWSConfig.__dataclass_fields__.keys()
    if missing or extra:
        raise ValueError(f"Invalid infra fields: missing={sorted(missing)}, unknown={sorted(extra)}")
    for field in ("name", "project", "region", "instance_type", "key_name", "ssh_user"):
        value = data.get(field, AWSConfig.__dataclass_fields__[field].default)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
            raise ValueError(f"{field} must be a non-empty name using letters, digits, _, . or -")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", data.get("ssh_user", "ubuntu")):
        raise ValueError("ssh_user must be an SSH login name")
    for field, pattern in (("subnet_id", r"subnet-[0-9a-f]+"),
                           ("ami_id", r"ami-[0-9a-f]+")):
        if field == "ami_id" and field not in data:
            continue
        if not isinstance(data[field], str) or not re.fullmatch(pattern, data[field]):
            raise ValueError(f"Invalid {field}")
    groups = data["security_group_ids"]
    if (not isinstance(groups, list) or not groups or
            any(not isinstance(group, str) or not re.fullmatch(r"sg-[0-9a-f]+", group)
                for group in groups) or len(set(groups)) != len(groups)):
        raise ValueError("security_group_ids must be a non-empty list of unique sg- IDs")
    if type(data["root_gb"]) is not int or not 8 <= data["root_gb"] <= 16384:
        raise ValueError("root_gb must be an integer between 8 and 16384")
    if bool(data.get("ami_id")) == bool(data.get("ami_ssm")):
        raise ValueError("Provide exactly one of ami_id or ami_ssm")
    for field in ("ami_ssm", "profile", "instance_profile"):
        if field in data and (not isinstance(data[field], str) or not data[field].strip()):
            raise ValueError(f"{field} must be a non-empty string")
    if data.get("market", "ondemand") not in ("ondemand", "spot"):
        raise ValueError("market must be ondemand or spot; there is no automatic fallback")
    if type(data.get("private_ip", False)) is not bool:
        raise ValueError("private_ip must be a boolean")
    for field in ("key", "known_hosts", "state_dir"):
        if field in data:
            data[field] = _resolve_path(data[field], path.resolve().parent, field)
            if data[field] is None:
                raise ValueError(f"{field} must be a path")
    return AWSConfig(**data)
