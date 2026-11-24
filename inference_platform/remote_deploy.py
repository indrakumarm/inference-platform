"""Self-contained remote deployment payload; requires only the host's Python 3."""
import csv
import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
CUDA_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"
CACHE_VOLUME = "inference-platform-huggingface"
IMAGE_RE = re.compile(r"(?:docker\.io/)?vllm/vllm-openai@sha256:[0-9a-f]{64}\Z")
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(network) for network in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
))
logger = logging.getLogger(__name__)


def run(command, timeout=900):
    logger.info("Running remote deployment command: %s", command[0])
    logger.debug("Deployment command arguments: %s", command)
    try:
        result = subprocess.run(command, check=True, timeout=timeout, text=True,
                                stdout=sys.stderr, stderr=sys.stderr)
    except subprocess.CalledProcessError as error:
        logger.error("Remote command %s exited with status %d",
                     command[0], error.returncode)
        if error.stderr:
            logger.debug("Remote command stderr: %s", error.stderr.strip())
        raise
    logger.debug("Remote command %s completed", command[0])
    return result


def probe(command):
    logger.debug("Running deployment preflight probe: %s", command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        logger.debug("Deployment probe exited %d: %s", result.returncode, command[0])
        return {"code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        logger.warning("Deployment probe failed: %s", error)
        return {"code": 127 if isinstance(error, FileNotFoundError) else 124,
                "stdout": "", "stderr": str(error)}


def validate_private_bind_ip(value):
    if value is None:
        return
    if isinstance(value, str):
        try:
            address = ipaddress.IPv4Address(value)
        except ipaddress.AddressValueError:
            pass
        else:
            if any(address in network for network in PRIVATE_NETWORKS):
                return
    raise ValueError("private_bind_ip must be a specific RFC 1918 IPv4 address or null")


def validate(profile):
    required = {
        "name", "image", "model", "model_revision", "tokenizer", "tokenizer_revision",
        "gpu_indices", "min_gpu_memory_gib", "min_free_disk_gib", "max_model_len",
        "dtype", "gpu_memory_utilization", "port", "shm_size_gib",
    }
    if not isinstance(profile, dict) or set(profile) - {"private_bind_ip"} != required:
        raise ValueError("Invalid deployment profile fields")
    validate_private_bind_ip(profile.get("private_bind_ip"))
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", profile["name"]):
        raise ValueError("Invalid deployment name")
    if not IMAGE_RE.fullmatch(profile["image"]):
        raise ValueError("Image must be pinned to the official vLLM repository digest")
    if any(not isinstance(profile[field], str) or not MODEL_RE.fullmatch(profile[field])
           for field in ("model", "tokenizer")):
        raise ValueError("Invalid public model identifier")
    if any(not isinstance(profile[field], str) or not REVISION_RE.fullmatch(profile[field])
           for field in ("model_revision", "tokenizer_revision")):
        raise ValueError("Model/tokenizer revisions must be commit hashes")
    indices = profile["gpu_indices"]
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Invalid GPU assignments")
    if profile["dtype"] not in ("auto", "bfloat16", "float16", "float32"):
        raise ValueError("Invalid dtype")
    if type(profile["port"]) is not int or not 1024 <= profile["port"] <= 65535:
        raise ValueError("Invalid endpoint port")


def docker_prefix():
    if os.geteuid() == 0:
        return []
    if probe(["sudo", "-n", "true"])["code"] == 0:
        return ["sudo", "-n", "--"]
    return None


def check_bind(host, port):
    try:
        with socket.socket() as listener:
            listener.bind((host, port))
    except OSError as error:
        return str(error)
    return ""


def gpu_facts():
    gpu = probe(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"])
    gpus = []
    if gpu["code"] == 0:
        try:
            for row in csv.reader(gpu["stdout"].splitlines(), skipinitialspace=True):
                if len(row) != 4:
                    raise ValueError("unexpected nvidia-smi GPU row")
                gpus.append({"index": int(row[0]), "name": row[1],
                             "total_memory_mib": int(row[2]), "free_memory_mib": int(row[3])})
        except ValueError as error:
            gpu["code"] = 1
            gpu["stderr"] = str(error)
            gpus = []
    processes = probe(["nvidia-smi", "--query-compute-apps=pid,process_name",
                       "--format=csv,noheader"])
    return {
        "gpu_error": (gpu["stderr"] or f"nvidia-smi exited {gpu['code']}") if gpu["code"] != 0 else "",
        "gpus": gpus,
        "gpu_process_error": (processes["stderr"] or f"nvidia-smi exited {processes['code']}")
        if processes["code"] != 0 else "",
        "gpu_processes": processes["stdout"].splitlines(),
    }


def deployment_inspect(profile):
    logger.info("Collecting deployment preflight facts")
    validate(profile)
    prefix = docker_prefix()
    info = probe((prefix or []) + DOCKER + ["info", "--format", "{{.ID}}"]) if prefix is not None else {
        "code": 1, "stdout": "", "stderr": "root or passwordless sudo is required"}
    gpu = gpu_facts()
    containers = probe((prefix or []) + DOCKER + ["ps", "--format", "{{.Names}}"]) if prefix is not None else {
        "code": 1, "stdout": "", "stderr": "root or passwordless sudo is required"}
    all_containers = probe((prefix or []) + DOCKER + ["ps", "-a", "--format", "{{.Names}}"]) if prefix is not None else {
        "code": 1, "stdout": "", "stderr": "root or passwordless sudo is required"}
    try:
        disk = os.statvfs("/")
        free_disk_bytes = disk.f_bavail * disk.f_frsize
    except OSError as error:
        free_disk_bytes = 0
        disk_error = str(error)
    else:
        disk_error = ""
    port_available = not check_bind("127.0.0.1", profile["port"])
    private_ip = profile.get("private_bind_ip")
    private_bind_error = check_bind(private_ip, profile["port"]) if private_ip else ""
    return {
        "docker_ready": info["code"] == 0,
        "docker_error": info["stderr"],
        **gpu,
        "container_error": (containers["stderr"] if containers["code"] != 0
                            else all_containers["stderr"] if all_containers["code"] != 0 else ""),
        "running_containers": containers["stdout"].splitlines(),
        "target_container_exists": (
            "inference-" + profile["name"] in all_containers["stdout"].splitlines()
            if all_containers["code"] == 0 else False
        ),
        "free_disk_bytes": free_disk_bytes,
        "disk_error": disk_error,
        "port_available": port_available,
        "private_bind_error": private_bind_error,
    }


def disk_space_blocker(facts, minimum_gib):
    if facts["disk_error"]:
        return "Cannot query free disk space: " + facts["disk_error"]
    free_bytes = facts["free_disk_bytes"]
    if free_bytes < minimum_gib * 1024**3:
        return (f"Free space on /: {free_bytes / 1024**3:.2f} GiB available; "
                f"{minimum_gib} GiB required (min_free_disk_gib).")
    return None


def port_blockers(facts, port, private_ip):
    result = []
    if not facts["port_available"]:
        result.append(f"Port {port} is unavailable on 127.0.0.1.")
    if private_ip and facts["private_bind_error"]:
        result.append(f"Cannot bind {private_ip}:{port}: {facts['private_bind_error']}")
    return result


def blockers(facts, profile, allow_target_container=False):
    result = []
    if not facts["docker_ready"]:
        result.append("Rootful local Docker is unavailable: " + facts["docker_error"])
    if facts["gpu_error"]:
        result.append("Cannot query NVIDIA GPUs: " + facts["gpu_error"])
    gpus = {gpu["index"]: gpu for gpu in facts["gpus"]}
    missing = [index for index in profile["gpu_indices"] if index not in gpus]
    if missing:
        result.append("Selected GPU indices are unavailable: " + ", ".join(map(str, missing)))
    for index in profile["gpu_indices"]:
        gpu = gpus.get(index)
        if gpu and gpu["free_memory_mib"] < profile["min_gpu_memory_gib"] * 1024:
            result.append(f"GPU {index} has less free memory than the profile minimum.")
    if facts["gpu_process_error"]:
        result.append("Cannot verify GPU compute processes: " + facts["gpu_process_error"])
    elif facts["gpu_processes"]:
        result.append("GPU compute processes are active; deployment requires an idle dedicated host.")
    if facts["container_error"]:
        result.append("Cannot enumerate running containers: " + facts["container_error"])
    elif facts["running_containers"]:
        result.append("Running containers exist; deployment requires a dedicated host with no active containers.")
    if facts["target_container_exists"] and not allow_target_container:
        result.append("A container for this deployment name already exists; inspect it before retrying.")
    disk_blocker = disk_space_blocker(facts, profile["min_free_disk_gib"])
    if disk_blocker:
        result.append(disk_blocker)
    result.extend(port_blockers(facts, profile["port"], profile.get("private_bind_ip")))
    return result


def require_ready(profile):
    facts = deployment_inspect(profile)
    blocked = blockers(facts, profile)
    if blocked:
        logger.warning("Deployment preflight blocked: %s", blocked)
        raise RuntimeError("Deployment preflight blocked: " + " ".join(blocked))


def check_model(profile):
    endpoint = f"http://127.0.0.1:{profile['port']}"
    with urllib.request.urlopen(endpoint + "/v1/models", timeout=10) as response:
        models = json.loads(response.read())
    ids = [entry["id"] for entry in models.get("data", [])]
    if profile["model"] not in ids:
        raise RuntimeError(f"Served model identity mismatch: expected {profile['model']}, got {ids}")
    request = urllib.request.Request(
        endpoint + "/v1/completions",
        data=json.dumps({"model": profile["model"], "prompt": "Reply with ready.",
                         "max_tokens": 8, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        generation = json.loads(response.read())
    text = generation["choices"][0]["text"]
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Generation smoke test returned empty text")
    return {"endpoint": endpoint + "/v1", "model": profile["model"],
            "generation_smoke_test": "passed"}


def wait_for_model(profile, container):
    endpoint = f"http://127.0.0.1:{profile['port']}"
    logger.info("Waiting for model API identity and generation checks at %s", endpoint)
    deadline = time.monotonic() + 1800
    next_progress = time.monotonic()
    last_error = "model API has not responded"
    while time.monotonic() < deadline:
        state = probe((docker_prefix() or []) + DOCKER + ["inspect", "--format", "{{.State.Running}}", container])
        if state["code"] != 0 or state["stdout"] != "true":
            logs = probe((docker_prefix() or []) + DOCKER + ["logs", "--tail", "80", container])
            raise RuntimeError("vLLM container exited before readiness: " +
                               (logs["stdout"] or logs["stderr"] or state["stderr"]))
        try:
            return check_model(profile)
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError) as error:
            last_error = str(error)
            logger.debug("Model readiness check not yet successful: %s", last_error)
            now = time.monotonic()
            if now >= next_progress:
                logs = probe((docker_prefix() or []) + DOCKER + [
                    "logs", "--tail", "20", container])
                if logs["code"] == 0 and logs["stdout"]:
                    logger.info("vLLM startup is ongoing; recent container logs:\n%s",
                                logs["stdout"])
                else:
                    logger.info("vLLM is still starting; latest API check: %s", last_error)
                next_progress = now + 30
            time.sleep(5)
    logs = probe((docker_prefix() or []) + DOCKER + ["logs", "--tail", "80", container])
    raise RuntimeError("vLLM did not become ready within 30 minutes: " + last_error +
                       "\n" + (logs["stdout"] or logs["stderr"]))


def deploy(profile):
    logger.info("Starting deployment %s using model %s", profile["name"], profile["model"])
    validate(profile)
    require_ready(profile)
    prefix = docker_prefix()
    docker = prefix + DOCKER
    logger.info("Pulling the pinned vLLM image")
    run(docker + ["pull", profile["image"]], timeout=1800)
    require_ready(profile)
    logger.info("Verifying passthrough for the assigned GPUs")
    run(docker + ["run", "--rm", "--network", "none", "--gpus",
                  "device=" + ",".join(map(str, profile["gpu_indices"])),
                  CUDA_IMAGE, "nvidia-smi", "-L"], timeout=300)
    require_ready(profile)
    container = "inference-" + profile["name"]
    run(docker + ["volume", "create", CACHE_VOLUME])
    publish = ["--publish", f"127.0.0.1:{profile['port']}:8000"]
    if profile.get("private_bind_ip"):
        publish.extend(["--publish", f"{profile['private_bind_ip']}:{profile['port']}:8000"])
    command = docker + [
        "run", "-d", "--name", container, "--restart", "unless-stopped",
        "--label", "com.inference-platform.managed=true",
        "--label", "com.inference-platform.profile=" + profile["name"],
        "--label", "com.inference-platform.model=" + profile["model"],
        "--label", "com.inference-platform.port=" + str(profile["port"]),
        "--gpus", "device=" + ",".join(map(str, profile["gpu_indices"])),
        *publish,
        "--shm-size", f"{profile['shm_size_gib']}g",
        "--volume", CACHE_VOLUME + ":/root/.cache/huggingface",
        "--env", "HF_HUB_DISABLE_TELEMETRY=1",
        profile["image"], "--model", profile["model"],
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", profile["model"],
        "--revision", profile["model_revision"],
        "--tokenizer", profile["tokenizer"],
        "--tokenizer-revision", profile["tokenizer_revision"],
        "--tensor-parallel-size", str(len(profile["gpu_indices"])),
        "--max-model-len", str(profile["max_model_len"]),
        "--dtype", profile["dtype"],
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
    ]
    run(command, timeout=300)
    readiness = wait_for_model(profile, container)
    logger.info("Deployment %s passed readiness checks", profile["name"])
    return readiness


def list_deployments():
    logger.info("Listing inference-platform containers")
    prefix = docker_prefix()
    if prefix is None:
        raise PermissionError("Root or passwordless sudo is required to list Docker containers")
    result = probe(prefix + DOCKER + ["ps", "-a", "--format", "{{json .}}"])
    if result["code"] != 0:
        raise RuntimeError("Cannot list Docker containers: " + result["stderr"])
    deployments = []
    for line in result["stdout"].splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("Docker returned invalid container-list JSON") from error
        name = row.get("Names")
        if not isinstance(name, str):
            raise ValueError("Docker container list omitted a container name")
        labels = {}
        for item in (row.get("Labels") or "").split(","):
            key, separator, value = item.partition("=")
            if separator:
                labels[key] = value
        is_labeled = labels.get("com.inference-platform.managed") == "true"
        if not is_labeled and not name.startswith("inference-"):
            continue
        legacy_profile = name.removeprefix("inference-") if name.startswith("inference-") else name
        deployments.append({
            "container": name,
            "profile": labels.get("com.inference-platform.profile", legacy_profile),
            "model": labels.get("com.inference-platform.model", "unknown (legacy container)"),
            "state": row.get("State", "unknown"),
            "status": row.get("Status", ""),
            "image": row.get("Image", ""),
            "ports": row.get("Ports", ""),
            "metadata_source": "labels" if is_labeled else "container-name",
        })
    return sorted(deployments, key=lambda deployment: deployment["container"])


def container_status(profile):
    validate(profile)
    prefix = docker_prefix()
    if prefix is None:
        raise PermissionError("Root or passwordless sudo is required to manage Docker containers")
    docker = prefix + DOCKER
    name = "inference-" + profile["name"]
    containers = probe(docker + ["ps", "-a", "--format", "{{.Names}}"])
    if containers["code"] != 0:
        raise RuntimeError("Cannot enumerate Docker containers: " + containers["stderr"])
    if name not in containers["stdout"].splitlines():
        return {"name": name, "exists": False, "state": "not-found"}
    result = probe(docker + ["inspect", "--format", "{{json .State}}", name])
    if result["code"] != 0:
        raise RuntimeError(f"Cannot inspect {name}: {result['stderr']}")
    try:
        state = json.loads(result["stdout"])
    except json.JSONDecodeError as error:
        raise ValueError(f"Docker returned invalid state for {name}") from error
    return {
        "name": name,
        "exists": True,
        "state": state["Status"],
        "running": state["Running"],
        "exit_code": state["ExitCode"],
        "error": state["Error"],
        "started_at": state["StartedAt"],
        "finished_at": state["FinishedAt"],
    }


def manage_container(operation, profile):
    if operation not in ("status", "start", "stop", "remove"):
        raise ValueError("Unsupported container lifecycle operation")
    validate(profile)
    before = container_status(profile)
    if operation == "status":
        return before
    if not before["exists"]:
        raise RuntimeError(f"{before['name']} does not exist; deploy the profile first")
    if operation == "start" and before["running"]:
        return {**before, "changed": False}
    if operation == "stop" and not before["running"]:
        return {**before, "changed": False}
    if operation == "remove" and before["running"]:
        raise RuntimeError(f"{before['name']} is running; stop it before removing it")

    prefix = docker_prefix()
    if prefix is None:
        raise PermissionError("Root or passwordless sudo is required to manage Docker containers")
    docker = prefix + DOCKER
    name = before["name"]
    if operation == "start":
        facts = deployment_inspect(profile)
        blocked = blockers(facts, profile, allow_target_container=True)
        if blocked:
            raise RuntimeError("Cannot start container: " + " ".join(blocked))
        run(docker + ["start", name])
    elif operation == "stop":
        run(docker + ["stop", "--time", "60", name])
    elif operation == "remove":
        run(docker + ["rm", name])
        return {"name": name, "exists": False, "state": "not-found",
                "removed": True}
    after = container_status(profile)
    return {**after, "changed": True}


def main():
    operation = sys.argv[1] if len(sys.argv) == 2 else ""
    profile = json.loads(sys.stdin.read())
    if operation == "list":
        print(json.dumps(list_deployments()))
    elif operation == "inspect":
        print(json.dumps(deployment_inspect(profile)))
    elif operation == "deploy":
        print(json.dumps(deploy(profile)))
    elif operation in ("status", "start", "stop", "remove"):
        print(json.dumps(manage_container(operation, profile)))
    else:
        raise SystemExit("Expected list, inspect, deploy, status, start, stop or remove")


if __name__ == "__main__":
    configure_logging()
    main()
