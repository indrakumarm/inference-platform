"""Detached, one-shot vLLM supervisor executed inside a Vast container."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


directory = Path(sys.argv[1])
token = sys.argv[2]
state_path = directory / "state.json"
lock_path = directory / "lock"


def update(**changes):
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        if state["token"] != token:
            raise RuntimeError("Deployment identity changed while supervisor was running")
        state.update(changes)
        fd, temporary = tempfile.mkstemp(prefix=".state-", dir=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, state_path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def proc_start(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    return stat[stat.rfind(")") + 2:].split()[19]


def main():
    state = json.loads(state_path.read_text())
    if state["token"] != token:
        raise RuntimeError("Deployment identity changed before supervisor startup")
    profile = state["profile"]
    environment = os.environ.copy()
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = list(map(str, profile["gpu_indices"]))
    else:
        devices = [item.strip() for item in visible.split(",")]
        if max(profile["gpu_indices"]) >= len(devices) or any(not item for item in devices):
            raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the selected GPU indices")
        selected = [devices[index] for index in profile["gpu_indices"]]
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    command = [
        "vllm", "serve", profile["model"], "--host", "127.0.0.1",
        "--port", str(profile["port"]), "--served-model-name", profile["model"],
        "--revision", profile["model_revision"], "--tokenizer", profile["tokenizer"],
        "--tokenizer-revision", profile["tokenizer_revision"],
        "--tensor-parallel-size", str(len(profile["gpu_indices"])),
        "--max-model-len", str(profile["max_model_len"]), "--dtype", profile["dtype"],
        "--gpu-memory-utilization", str(profile["gpu_memory_utilization"]),
    ]
    child = None
    requested_stop = False
    stop_deadline = None

    def stop(_signal, _frame):
        nonlocal requested_stop, stop_deadline
        requested_stop = True
        stop_deadline = time.monotonic() + 60
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    try:
        with (directory / "vllm.log").open("a") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=environment,
                                     start_new_session=True)
            update(phase="running", pid=os.getpid(), starttime=proc_start(os.getpid()),
                   started_at=time.time())
            if requested_stop:
                os.killpg(child.pid, signal.SIGTERM)
            while True:
                try:
                    exit_code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if requested_stop and time.monotonic() >= stop_deadline:
                        os.killpg(child.pid, signal.SIGKILL)
                        exit_code = child.wait()
                        break
        deadline = time.monotonic() + 5
        while True:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                phase = "exited"
                error = "" if requested_stop or exit_code == 0 else "vLLM exited; inspect vllm.log"
                break
            if time.monotonic() >= deadline:
                phase = "orphaned"
                error = "vLLM subprocesses remain; inspect the process group manually"
                break
            time.sleep(0.25)
        update(phase=phase, exit_code=exit_code, finished_at=time.time(), error=error)
    except (OSError, RuntimeError, ValueError) as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            except ProcessLookupError:
                child.wait(timeout=5)
        update(phase="exited", exit_code=None, finished_at=time.time(), error=str(error))
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        update(phase="orphaned", error=f"Supervisor failed: {error}",
               finished_at=time.time())
        raise
