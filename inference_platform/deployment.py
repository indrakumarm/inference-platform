import json
from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import re

from .remote_deploy import disk_space_blocker, port_blockers, validate_private_bind_ip


logger = logging.getLogger(__name__)


_PROFILE_FIELDS = {
    "name", "image", "model", "model_revision", "tokenizer", "tokenizer_revision",
    "gpu_indices", "min_gpu_memory_gib", "min_free_disk_gib", "max_model_len",
    "dtype", "gpu_memory_utilization", "port", "shm_size_gib",
}
_IMAGE = re.compile(r"(?:docker\.io/)?vllm/vllm-openai@sha256:[0-9a-f]{64}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class DeploymentProfile:
    name: str
    image: str
    model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    gpu_indices: list[int]
    min_gpu_memory_gib: float
    min_free_disk_gib: float
    max_model_len: int
    dtype: str
    gpu_memory_utilization: float
    port: int
    shm_size_gib: int
    private_bind_ip: str | None = None

    def as_dict(self):
        return asdict(self)


def parse_profile(data):
    logger.debug("Validating deployment profile")
    if not isinstance(data, dict):
        raise ValueError("Deployment profile must be a JSON object")
    missing = _PROFILE_FIELDS - data.keys()
    extra = data.keys() - _PROFILE_FIELDS - {"private_bind_ip"}
    if missing or extra:
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown fields: " + ", ".join(sorted(extra)))
        raise ValueError("Invalid deployment profile (" + "; ".join(details) + ")")
    if not isinstance(data["name"], str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", data["name"]):
        raise ValueError("name must be 1-40 lowercase letters, digits or hyphens, starting with a letter")
    if not isinstance(data["image"], str) or not _IMAGE.fullmatch(data["image"]):
        raise ValueError("image must pin the official vllm/vllm-openai image by sha256 digest")
    for field in ("model", "tokenizer"):
        if not isinstance(data[field], str) or not _MODEL.fullmatch(data[field]):
            raise ValueError(f"{field} must be a public Hugging Face model identifier")
    for field in ("model_revision", "tokenizer_revision"):
        if not isinstance(data[field], str) or not _REVISION.fullmatch(data[field]):
            raise ValueError(f"{field} must be a 40-character lowercase commit hash")
    indices = data["gpu_indices"]
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("gpu_indices must be a non-empty list of unique non-negative integers")
    for field, low, high in (
        ("min_gpu_memory_gib", 1, 1024),
        ("min_free_disk_gib", 1, 100000),
    ):
        value = data[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f"{field} must be between {low} and {high} GiB")
    if type(data["max_model_len"]) is not int or not 128 <= data["max_model_len"] <= 131072:
        raise ValueError("max_model_len must be an integer between 128 and 131072 tokens")
    if data["dtype"] not in ("auto", "bfloat16", "float16", "float32"):
        raise ValueError("dtype must be auto, bfloat16, float16 or float32")
    utilization = data["gpu_memory_utilization"]
    if (isinstance(utilization, bool) or not isinstance(utilization, (int, float))
            or not 0.1 <= utilization <= 0.95):
        raise ValueError("gpu_memory_utilization must be between 0.1 and 0.95")
    if type(data["port"]) is not int or not 1024 <= data["port"] <= 65535:
        raise ValueError("port must be an integer between 1024 and 65535")
    if type(data["shm_size_gib"]) is not int or not 1 <= data["shm_size_gib"] <= 128:
        raise ValueError("shm_size_gib must be an integer between 1 and 128")
    validate_private_bind_ip(data.get("private_bind_ip"))
    return DeploymentProfile(**data)


def load_profile(path):
    logger.info("Loading deployment profile from %s", path)
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid deployment profile JSON: {error}") from error
    return parse_profile(data)


def deployment_blockers(facts, profile):
    blockers = []
    if not facts["docker_ready"]:
        blockers.append("Rootful local Docker is unavailable.")
    if facts["gpu_error"]:
        blockers.append("Cannot query NVIDIA GPUs: " + facts["gpu_error"])
    gpus = {gpu["index"]: gpu for gpu in facts["gpus"]}
    missing = [index for index in profile.gpu_indices if index not in gpus]
    if missing:
        blockers.append("Selected GPU indices are unavailable: " + ", ".join(map(str, missing)))
    for index in profile.gpu_indices:
        gpu = gpus.get(index)
        if gpu and gpu["free_memory_mib"] < profile.min_gpu_memory_gib * 1024:
            blockers.append(
                f"GPU {index} has less free memory than the profile minimum "
                f"({gpu['free_memory_mib'] / 1024:.1f} < {profile.min_gpu_memory_gib:g} GiB)."
            )
    if facts["gpu_process_error"]:
        blockers.append("Cannot verify GPU compute processes: " + facts["gpu_process_error"])
    elif facts["gpu_processes"]:
        blockers.append("GPU compute processes are active; deployment requires an idle dedicated host.")
    if facts["container_error"]:
        blockers.append("Cannot enumerate running containers: " + facts["container_error"])
    elif facts["running_containers"]:
        blockers.append("Running containers exist; deployment requires a dedicated host with no active containers.")
    if facts["target_container_exists"]:
        blockers.append("A container for this deployment name already exists; inspect it before retrying.")
    disk_blocker = disk_space_blocker(facts, profile.min_free_disk_gib)
    if disk_blocker:
        blockers.append(disk_blocker)
    blockers.extend(port_blockers(facts, profile.port, profile.private_bind_ip))
    logger.debug("Deployment preflight produced %d blocker(s)", len(blockers))
    return blockers
