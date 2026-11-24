"""Self-contained Vast container workload operations; no host packages are installed."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile
import time
import urllib.error
import uuid

if __package__:
    from .logging_config import configure_logging
    from . import remote_deploy as common
else:
    from logging_config import configure_logging
    import remote_deploy as common


ROOT = Path.home() / ".local" / "state" / "inference-platform" / "vast"

TORCH_PROBE = """
import json, torch
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
gpus = []
for index in range(count):
    free, total = torch.cuda.mem_get_info(index)
    gpus.append({"index": index, "name": torch.cuda.get_device_name(index),
                 "free_memory_mib": free // 1048576, "total_memory_mib": total // 1048576})
print(json.dumps({"available": available, "count": count, "gpus": gpus}))
"""


def torch_probe():
    return common.probe([sys.executable, "-c", TORCH_PROBE])


def cache_probe():
    return common.probe([sys.executable, "-c",
                         "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)"])


def existing_ancestor(path):
    while True:
        try:
            path.stat()
            return path
        except FileNotFoundError:
            if path.parent == path:
                raise
            path = path.parent


def proc_fields(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rfind(")") + 2:].split()


def container_epoch():
    return [Path("/proc/sys/kernel/random/boot_id").read_text().strip(), proc_fields(1)[19]]


def supervisor_alive(state):
    pid = state.get("pid")
    if type(pid) is not int or pid <= 1:
        return False
    try:
        fields = proc_fields(pid)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return (fields[0] != "Z" and fields[19] == state.get("starttime")
            and state["token"].encode() in command)


def directory(profile):
    common.validate(profile)
    return ROOT / profile["name"]


@contextmanager
def locked(path):
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    with (path / "lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def write_state(path, state):
    fd, temporary = tempfile.mkstemp(prefix=".state-", dir=path)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path / "state.json")
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_state(path):
    try:
        state = json.loads((path / "state.json").read_text())
    except FileNotFoundError:
        return None
    if (not isinstance(state, dict) or state.get("phase") not in ("starting", "running", "exited", "orphaned")
            or not isinstance(state.get("profile"), dict)
            or not isinstance(state.get("token"), str)
            or not isinstance(state.get("epoch"), list)):
        raise ValueError(f"Invalid Vast workload state at {path}; inspect manually")
    common.validate(state["profile"])
    return state


def state_status(state):
    if state is None:
        return {"name": "", "exists": False, "state": "not-found", "running": False}
    phase = state["phase"]
    error = state.get("error", "")
    if phase in ("starting", "running"):
        if state["epoch"] != container_epoch():
            phase = "exited"
            error = "The container restarted; start the model manually."
        elif phase == "running" and supervisor_alive(state):
            pass
        elif phase == "starting":
            requested_at = state.get("requested_at")
            if not isinstance(requested_at, (int, float)) or time.time() - requested_at > 30:
                phase = "unknown"
        else:
            phase = "unknown"
    return {
        "name": "inference-" + state["profile"]["name"], "exists": True,
        "state": phase, "running": phase in ("starting", "running"),
        "exit_code": state.get("exit_code"), "error": error,
        "started_at": state.get("started_at"), "finished_at": state.get("finished_at"),
    }


def status(profile):
    path = directory(profile)
    if not path.exists():
        return {"name": "inference-" + profile["name"], "exists": False,
                "state": "not-found", "running": False}
    with locked(path):
        state = read_state(path)
        if state is not None and state["profile"] != profile:
            raise RuntimeError("Deployment profile differs from the saved profile; use the original profile")
        result = state_status(state)
        if not result["exists"]:
            result["name"] = "inference-" + profile["name"]
        return result


def host_inspect():
    gpu = common.probe(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                        "--format=csv,noheader"])
    processes = common.probe(["nvidia-smi", "--query-compute-apps=pid,process_name",
                              "--format=csv,noheader"])
    torch = torch_probe()
    cache = cache_probe()
    executable = shutil.which("vllm")
    vllm = {"code": 0 if executable else 127, "stdout": executable or "",
            "stderr": "" if executable else "vllm executable not found in PATH"}
    return {"schema": 1, "runtime": "vast-container",
            "os": {"PRETTY_NAME": "Vast container"}, "architecture": os.uname().machine,
            "root": os.geteuid() == 0, "privileged": False,
            "root_free_bytes": shutil.disk_usage("/").free,
            "cache_free_bytes": shutil.disk_usage(Path.home()).free,
            "checks": {"gpu": gpu, "gpu_processes": processes, "torch": torch,
                       "vllm": vllm, "cache": cache}}


def inspect_deployment(profile):
    common.validate(profile)
    gpu = common.gpu_facts()
    torch = torch_probe()
    torch_error = ""
    torch_gpus = []
    if torch["code"] != 0:
        torch_error = torch["stderr"] or f"PyTorch GPU probe exited {torch['code']}"
    else:
        try:
            torch_data = json.loads(torch["stdout"])
            torch_gpus = torch_data["gpus"]
            if (not isinstance(torch_gpus, list) or
                    any(not isinstance(item, dict)
                        or type(item.get("index")) is not int
                        or not isinstance(item.get("name"), str)
                        or type(item.get("free_memory_mib")) is not int
                        or item["free_memory_mib"] < 0 for item in torch_gpus)):
                raise ValueError("invalid GPU list")
        except (ValueError, KeyError, TypeError) as error:
            torch_error = f"Invalid PyTorch GPU probe response: {error}"
    cache = cache_probe()
    try:
        if cache["code"] != 0 or not cache["stdout"].strip():
            raise RuntimeError(cache["stderr"] or "Hugging Face cache path probe failed")
        cache_path = Path(cache["stdout"].strip()).expanduser()
        if not cache_path.is_absolute():
            raise ValueError("Hugging Face cache path must be absolute")
        disk = os.statvfs(existing_ancestor(cache_path))
        free_bytes = disk.f_bavail * disk.f_frsize
        disk_error = ""
    except (OSError, ValueError, RuntimeError) as error:
        free_bytes = 0
        disk_error = str(error)
    try:
        shm = os.statvfs("/dev/shm")
        shm_bytes = shm.f_bavail * shm.f_frsize
        shm_error = ""
    except OSError as error:
        shm_bytes = 0
        shm_error = str(error)
    return {**gpu, "gpus": torch_gpus, "torch_error": torch_error,
            "free_disk_bytes": free_bytes, "disk_error": disk_error,
            "shm_free_bytes": shm_bytes, "shm_error": shm_error,
            "port_available": not common.check_bind("127.0.0.1", profile["port"]),
            "private_bind_error": "", "target_exists": status(profile)["exists"]}


def blockers(facts, profile, allow_target=False):
    blocked = []
    if profile.get("private_bind_ip"):
        blocked.append("Vast workloads currently support loopback binding only; remove private_bind_ip.")
    if facts["gpu_error"]:
        blocked.append("Cannot query NVIDIA GPUs: " + facts["gpu_error"])
    if facts["torch_error"]:
        blocked.append("Cannot query PyTorch GPUs: " + facts["torch_error"])
    gpus = {gpu["index"]: gpu for gpu in facts["gpus"]}
    for index in profile["gpu_indices"]:
        gpu = gpus.get(index)
        if gpu is None:
            blocked.append(f"Selected GPU {index} is unavailable.")
        elif gpu["free_memory_mib"] < profile["min_gpu_memory_gib"] * 1024:
            blocked.append(f"GPU {index} has less free memory than the profile minimum.")
    if facts["gpu_process_error"]:
        blocked.append("Cannot verify GPU compute processes: " + facts["gpu_process_error"])
    elif facts["gpu_processes"]:
        blocked.append("GPU compute processes are active; deployment requires an idle container.")
    if facts["target_exists"] and not allow_target:
        blocked.append("A deployment with this name already exists; inspect it before retrying.")
    disk = common.disk_space_blocker(facts, profile["min_free_disk_gib"])
    if disk:
        blocked.append(disk.replace("Free space on /:", "Free space on the Hugging Face cache filesystem:"))
    if facts["shm_error"]:
        blocked.append("Cannot query /dev/shm: " + facts["shm_error"])
    elif facts["shm_free_bytes"] < profile["shm_size_gib"] * 1024**3:
        blocked.append(f"Available /dev/shm is less than {profile['shm_size_gib']} GiB. "
                       "Vast assigns shared memory when the container is created.")
    blocked.extend(common.port_blockers(facts, profile["port"], None))
    return blocked


def require_ready(profile, allow_target=False):
    host = host_inspect()
    for name in ("gpu", "gpu_processes", "torch", "vllm", "cache"):
        check = host["checks"][name]
        if check["code"] != 0:
            raise RuntimeError(f"{name} check failed: {check['stderr'] or check['stdout']}")
    try:
        torch = json.loads(host["checks"]["torch"]["stdout"])
    except (ValueError, TypeError) as error:
        raise RuntimeError("Invalid PyTorch GPU probe response") from error
    if (not isinstance(torch, dict) or torch.get("available") is not True
            or type(torch.get("count")) is not int or torch["count"] <= max(profile["gpu_indices"])):
        raise RuntimeError("PyTorch cannot access the selected GPU indices")
    blocked = blockers(inspect_deployment(profile), profile, allow_target)
    if blocked:
        raise RuntimeError("Deployment preflight blocked: " + " ".join(blocked))


def tail(path):
    try:
        with (path / "vllm.log").open(errors="replace") as stream:
            from collections import deque
            return "".join(deque(stream, maxlen=80))
    except FileNotFoundError:
        return ""


def start_process(profile, source, create):
    if not isinstance(source, str) or not source:
        raise ValueError("Missing supervisor program")
    path = directory(profile)
    with locked(ROOT):
        require_ready(profile, allow_target=not create)
        active = [item["profile"] for item in list_deployments() if item["state"] != "exited"]
        if active:
            raise RuntimeError("Another deployment is active or has uncertain state: " + ", ".join(active))
        with locked(path):
            existing = read_state(path)
            current = state_status(existing)
            if create and existing is not None:
                raise RuntimeError("Deployment already exists")
            if not create:
                if existing is None or existing["profile"] != profile:
                    raise RuntimeError("The original deployment profile is required to start")
                if current["state"] != "exited":
                    raise RuntimeError("Cannot start: deployment state is " + current["state"])
            state = {"profile": profile, "phase": "starting", "token": uuid.uuid4().hex,
                     "epoch": container_epoch(), "pid": None, "starttime": None,
                     "requested_at": time.time(), "exit_code": None, "error": "",
                     "started_at": None, "finished_at": None}
            write_state(path, state)
            try:
                with (path / "supervisor.log").open("a") as log, open(os.devnull, "rb") as null:
                    actions = [(os.POSIX_SPAWN_DUP2, null.fileno(), 0),
                               (os.POSIX_SPAWN_DUP2, log.fileno(), 1),
                               (os.POSIX_SPAWN_DUP2, log.fileno(), 2)]
                    os.posix_spawn(sys.executable,
                                   [sys.executable, "-c", source, str(path), state["token"]],
                                   os.environ.copy(), file_actions=actions, setsid=True)
            except (OSError, NotImplementedError, TypeError) as error:
                state.update(phase="exited", error=str(error), finished_at=time.time())
                write_state(path, state)
                raise RuntimeError(f"Cannot start detached supervisor: {error}") from error
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        current = status(profile)
        if current["state"] == "running":
            return
        if current["state"] != "starting":
            raise RuntimeError("vLLM supervisor failed to start: " + str(current["error"]) + "\n" + tail(path))
        time.sleep(0.2)
    raise RuntimeError("vLLM supervisor startup is uncertain; inspect status and supervisor.log before retrying")


def wait_for_model(profile):
    path = directory(profile)
    deadline = time.monotonic() + 1800
    next_progress = time.monotonic()
    last_error = "model API has not responded"
    while time.monotonic() < deadline:
        current = status(profile)
        if current["state"] != "running":
            raise RuntimeError("vLLM stopped before readiness: " + current["state"] + "\n" + tail(path))
        try:
            return common.check_model(profile)
        except RuntimeError:
            raise
        except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError) as error:
            last_error = str(error)
            if time.monotonic() >= next_progress:
                print("vLLM is starting; latest API check: " + last_error, file=sys.stderr, flush=True)
                next_progress = time.monotonic() + 30
            time.sleep(5)
    raise RuntimeError("vLLM did not become ready within 30 minutes: " + last_error + "\n" + tail(path))


def manage(operation, profile, source=None):
    path = directory(profile)
    if operation == "status":
        return status(profile)
    if operation == "start":
        start_process(profile, source, create=False)
        wait_for_model(profile)
        return status(profile)
    with locked(path):
        state = read_state(path)
        if state is None or state["profile"] != profile:
            raise RuntimeError("The original deployment profile is required")
        current = state_status(state)
        if operation == "stop":
            if current["state"] != "running":
                raise RuntimeError("Cannot stop safely: deployment state is " + current["state"])
            if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
                raise RuntimeError("Safe process signalling requires Linux pidfd support")
            try:
                pidfd = os.pidfd_open(state["pid"])
            except ProcessLookupError as error:
                raise RuntimeError("Supervisor exited before stop; inspect status") from error
            try:
                if not supervisor_alive(state):
                    raise RuntimeError("Supervisor identity changed; refusing to signal")
                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            finally:
                os.close(pidfd)
        elif operation == "remove":
            if current["state"] != "exited":
                raise RuntimeError("Cannot remove safely: deployment state is " + current["state"])
            (path / "state.json").unlink()
            return {"name": current["name"], "exists": False, "state": "not-found"}
        else:
            raise ValueError("Unsupported Vast workload operation")
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        current = status(profile)
        if current["state"] == "exited":
            return current
        if current["state"] not in ("running", "starting"):
            raise RuntimeError("Stop outcome is uncertain: " + current["state"])
        time.sleep(0.5)
    raise RuntimeError("Timed out stopping vLLM; inspect status before retrying")


def list_deployments():
    if not ROOT.exists():
        return []
    result = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir() or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", path.name):
            continue
        with locked(path):
            state = read_state(path)
            if state is None:
                continue
            if state["profile"]["name"] != path.name:
                raise ValueError(f"Deployment name does not match state directory: {path}")
            current = state_status(state)
            profile = state["profile"]
            result.append({"container": current["name"], "profile": profile["name"],
                           "model": profile["model"], "state": current["state"],
                           "status": "", "image": profile["image"],
                           "ports": f"127.0.0.1:{profile['port']}", "metadata_source": "process"})
    return result


def main():
    operation = sys.argv[1] if len(sys.argv) == 2 else ""
    payload = json.loads(sys.stdin.read())
    profile = payload.get("profile", {})
    if operation == "host-inspect":
        result = host_inspect()
    elif operation == "inspect":
        result = inspect_deployment(profile)
    elif operation == "deploy":
        start_process(profile, payload.get("supervisor"), create=True)
        result = wait_for_model(profile)
    elif operation == "list":
        result = list_deployments()
    elif operation in ("status", "start", "stop", "remove"):
        result = manage(operation, profile, payload.get("supervisor"))
    else:
        raise ValueError("Unsupported Vast workload operation")
    print(json.dumps(result))


if __name__ == "__main__":
    configure_logging()
    main()
