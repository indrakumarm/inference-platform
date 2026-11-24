"""Shared, non-secret configuration for gateway workflows."""

from dataclasses import dataclass
import json
from pathlib import Path

from .deployment import DeploymentProfile, load_profile, parse_profile
from .providers import Target, load_target, parse_target


@dataclass(frozen=True)
class PlatformConfig:
    target: Target
    deployment: DeploymentProfile
    gateway: dict
    env_file: Path


def _path(value, base, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_platform_config(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid platform JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Platform config must be a JSON object")
    if set(data) - {"target", "deployment", "gateway"} or not {"target", "deployment"} <= data.keys():
        raise ValueError("Platform config requires target and deployment; gateway is optional")
    base = path.resolve().parent
    target = data["target"]
    target = load_target(_path(target, base, "target")) if isinstance(target, str) else parse_target(target, base)
    deployment = data["deployment"]
    deployment = (load_profile(_path(deployment, base, "deployment"))
                  if isinstance(deployment, str) else parse_profile(deployment))
    gateway = data.get("gateway", {})
    if not isinstance(gateway, dict) or set(gateway) - {
            "mode", "listen", "tls", "ca_file", "client_port", "fetch_lab_certificate"}:
        raise ValueError("Invalid gateway configuration fields")
    gateway = dict(gateway)
    if "mode" in gateway and gateway["mode"] not in ("http", "https"):
        raise ValueError("gateway.mode must be http or https")
    if "listen" in gateway and (not isinstance(gateway["listen"], dict)
                                or set(gateway["listen"]) - {"host", "port"}):
        raise ValueError("gateway.listen accepts host and port only")
    if "tls" in gateway and (not isinstance(gateway["tls"], dict)
                             or set(gateway["tls"]) - {"server_name"}):
        raise ValueError("gateway.tls accepts server_name only")
    if "fetch_lab_certificate" in gateway and type(gateway["fetch_lab_certificate"]) is not bool:
        raise ValueError("gateway.fetch_lab_certificate must be a boolean")
    if gateway.get("ca_file") is not None:
        gateway["ca_file"] = _path(gateway["ca_file"], base, "gateway.ca_file")
    if gateway.get("fetch_lab_certificate") and not gateway.get("ca_file"):
        raise ValueError("gateway.fetch_lab_certificate requires gateway.ca_file")
    if gateway.get("mode") == "http" and (gateway.get("tls") or gateway.get("ca_file")
                                           or gateway.get("fetch_lab_certificate")):
        raise ValueError("HTTP gateway cannot have TLS settings")
    return PlatformConfig(target, deployment, gateway, base / ".env")


def load_env_value(path, name, *, required_file=False):
    """Read one variable without evaluating shell expressions or changing the environment."""
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        if required_file:
            raise
        return None
    token = None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator or not key.strip().isidentifier():
            raise ValueError(f"Invalid .env assignment on line {number}")
        if key.strip() != name:
            continue
        if token is not None:
            raise ValueError(f"Duplicate {name} in .env")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        token = value
    return token


def load_api_token(path, *, required_file=False):
    return load_env_value(path, "API_TOKEN", required_file=required_file)
