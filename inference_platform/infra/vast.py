"""Vast.ai container-instance provisioning; model lifecycle is separate."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
import uuid

from ..deployment import load_profile
from ..platform_config import load_env_value
from ..providers.base import _resolve_path
from .state import write_json


@dataclass(frozen=True)
class VastConfig:
    name: str
    image: str
    disk_gb: int
    gpu_name: str
    num_gpus: int
    key: str | None
    known_hosts: str | None
    state_dir: str
    env_file: Path


def load_vast_config(path):
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Vast config JSON: {error}") from error
    required = {"name", "deployment", "disk_gb", "gpu_name", "num_gpus"}
    optional = {"key", "known_hosts", "state_dir"}
    if not isinstance(data, dict) or required - data.keys() or data.keys() - required - optional:
        raise ValueError("Vast config requires name, deployment, disk_gb, gpu_name, num_gpus; "
                         "optional fields are key, known_hosts and state_dir")
    if not isinstance(data["name"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", data["name"]):
        raise ValueError("Invalid Vast resource name")
    if not isinstance(data["gpu_name"], str) or not data["gpu_name"].strip():
        raise ValueError("gpu_name must be a non-empty Vast GPU name")
    if type(data["num_gpus"]) is not int or data["num_gpus"] < 1:
        raise ValueError("num_gpus must be a positive integer")
    if type(data["disk_gb"]) is not int or data["disk_gb"] < 8:
        raise ValueError("disk_gb must be an integer of at least 8")
    deployment = _resolve_path(data["deployment"], path.parent, "deployment")
    if deployment is None:
        raise ValueError("deployment must be a path")
    profile = load_profile(deployment)
    if len(profile.gpu_indices) != data["num_gpus"]:
        raise ValueError("num_gpus must match the deployment profile's GPU count")
    key = _resolve_path(data.get("key"), path.parent, "key")
    known_hosts = _resolve_path(data.get("known_hosts"), path.parent, "known_hosts")
    state_dir = _resolve_path(data.get("state_dir", "~/.local/state/inference-platform/infra"),
                              path.parent, "state_dir")
    if state_dir is None:
        raise ValueError("state_dir must be a path")
    return VastConfig(data["name"], profile.image, data["disk_gb"], data["gpu_name"],
                      data["num_gpus"], key, known_hosts, state_dir, path.parent / ".env")


class VastAPI:
    def __init__(self, token):
        if not token or "\n" in token or "\r" in token:
            raise ValueError("Set VAST_API_KEY in the environment or the config directory's .env")
        self.token = token

    def request(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            "https://console.vast.ai/api/v0/" + path,
            data=data, method=method,
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Vast API {method} {path} failed with HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Vast API {method} {path} failed; outcome may be unknown") from error
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(f"Vast API {method} {path} returned invalid JSON") from error
        if not isinstance(result, dict) or result.get("success") is False:
            raise RuntimeError(f"Vast API {method} {path} returned an unsuccessful response")
        return result


def create_vast(config):
    token = os.environ.get("VAST_API_KEY")
    if token is None:
        token = load_env_value(config.env_file, "VAST_API_KEY")
    return VastInfra(config, VastAPI(token))


class VastInfra:
    def __init__(self, config, api):
        self.config = config
        self.api = api
        digest = hashlib.sha256(config.name.encode()).hexdigest()[:24]
        self.directory = Path(config.state_dir) / "vastai" / digest
        self.state_path = self.directory / "instance.json"
        self.target_path = self.directory / "target.json"

    def read_state(self):
        if not self.state_path.exists():
            return None
        data = json.loads(self.state_path.read_text())
        if (not isinstance(data, dict) or set(data) != {"name", "label", "offer_id", "instance_id", "phase"}
                or data["name"] != self.config.name
                or not isinstance(data["label"], str)
                or type(data["offer_id"]) is not int
                or (data["instance_id"] is not None and type(data["instance_id"]) is not int)
                or data["phase"] not in ("creating", "created", "destroyed")):
            raise ValueError("Invalid Vast state; reconcile manually before retrying")
        return data

    def search(self):
        result = self.api.request("POST", "bundles/", {
            "gpu_name": {"in": [self.config.gpu_name]},
            "num_gpus": {"eq": self.config.num_gpus},
            "verified": {"eq": True}, "rentable": {"eq": True},
            "type": "ondemand", "limit": 50,
        })
        offers = result.get("offers")
        if not isinstance(offers, list) or any(not isinstance(offer, dict) for offer in offers):
            raise RuntimeError("Vast offer response did not contain an offers list")
        return offers

    def status(self, state):
        if not state or state["phase"] == "destroyed":
            return None
        if state["instance_id"] is None:
            raise RuntimeError("Vast creation outcome is unknown. Check the Vast console and "
                               "reconcile the saved state manually; do not retry creation.")
        instance = self.api.request("GET", f"instances/{state['instance_id']}").get("instances")
        if (not isinstance(instance, dict) or instance.get("id") != state["instance_id"]
                or instance.get("label") != state["label"]):
            raise RuntimeError("Vast instance ID or ownership label does not match local state")
        return instance

    def create(self, offer_id):
        state = self.read_state()
        if state and state["phase"] != "destroyed":
            raise RuntimeError("A Vast instance or uncertain launch is already recorded; check status")
        state = {"name": self.config.name, "label": "inference-platform-" + uuid.uuid4().hex,
                 "offer_id": offer_id, "instance_id": None, "phase": "creating"}
        write_json(self.state_path, state)
        result = self.api.request("PUT", f"asks/{offer_id}/", {
            "image": self.config.image, "disk": self.config.disk_gb,
            "label": state["label"], "runtype": "ssh_proxy",
        })
        instance_id = result.get("new_contract")
        if type(instance_id) is not int or instance_id <= 0:
            raise RuntimeError("Vast did not return an instance ID; creation outcome is unknown")
        state.update(instance_id=instance_id, phase="created")
        write_json(self.state_path, state)
        return instance_id

    def change(self, operation, state, instance):
        latest = self.status(state)
        if latest.get("actual_status") != instance.get("actual_status"):
            raise RuntimeError("Vast instance state changed after confirmation; retry")
        path = f"instances/{state['instance_id']}"
        if operation == "down":
            response = self.api.request("DELETE", path)
        else:
            response = self.api.request("PUT", path, {"state": "running" if operation == "start" else "stopped"})
        if response.get("success") is not True:
            raise RuntimeError("Vast did not confirm the requested instance operation")
        if operation == "down":
            state["phase"] = "destroyed"
            write_json(self.state_path, state)

    def write_target(self, instance, destination):
        if instance.get("actual_status") != "running":
            raise RuntimeError("Vast instance is not running; try resolve after it starts")
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        if (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host)
                or type(port) is not int or not 1 <= port <= 65535):
            raise RuntimeError("Vast has not reported valid SSH connection details")
        data = {"provider": "vastai", "host": "root@" + host, "port": port}
        if self.config.key:
            data["key"] = self.config.key
        if self.config.known_hosts:
            data["known_hosts"] = self.config.known_hosts
        path = destination or self.target_path
        write_json(path, data)
        return path
