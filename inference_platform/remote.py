"""Self-contained remote payload: requires Python 3, not an installed package."""
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
CUDA_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"
logger = logging.getLogger(__name__)


def probe(command):
    logger.debug("Running host probe: %s", command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        logger.debug("Host probe exited %d: %s", result.returncode, command[0])
        return {"code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except FileNotFoundError as error:
        logger.debug("Host probe command not found: %s", command[0])
        return {"code": 127, "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired:
        logger.warning("Host probe timed out: %s", command[0])
        return {"code": 124, "stdout": "", "stderr": "Probe timed out after 20 seconds"}


def os_release():
    result = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key in ("ID", "VERSION_ID", "PRETTY_NAME"):
            result[key] = value.strip("\"'")
    return result


def inspect():
    logger.info("Collecting read-only host inspection facts")
    root = os.geteuid() == 0
    privileged = root or probe(["sudo", "-n", "true"])["code"] == 0
    prefix = [] if root or not privileged else ["sudo", "-n", "--"]
    checks = {
        "gpu": probe(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                      "--format=csv,noheader"]),
        "gpu_processes": probe(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                "--format=csv,noheader"]),
        "systemd": probe(["test", "-d", "/run/systemd/system"]),
        "docker_info": probe(prefix + DOCKER + ["info", "--format",
                                                "{{range $key, $value := .Runtimes}}{{$key}} {{end}}"]),
        "docker_active": probe(["systemctl", "is-active", "docker"]),
        "containers": probe(prefix + DOCKER + ["ps", "--format", "{{.ID}} {{.Names}}"]),
    }
    return {"schema": 1, "os": os_release(), "architecture": os.uname().machine,
            "root": root, "privileged": privileged,
            "root_free_bytes": shutil.disk_usage("/").free,
            "cache_free_bytes": shutil.disk_usage(Path.home()).free,
            "docker_installed": shutil.which("docker") is not None,
            "toolkit_installed": shutil.which("nvidia-ctk") is not None,
            "checks": checks}


def run(command):
    logger.info("Running host preparation command: %s", command[0])
    logger.debug("Preparation command arguments: %s", command)
    subprocess.run(command, check=True, timeout=900)


def ensure_idle():
    logger.debug("Rechecking for active GPU and Docker workloads")
    gpu = probe(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if gpu["code"] != 0 or gpu["stdout"]:
        raise RuntimeError("GPU workload check failed or GPU processes appeared; preparation stopped")
    info = probe(DOCKER + ["info"])
    if info["code"] == 0:
        containers = probe(DOCKER + ["ps", "-q"])
        if containers["code"] != 0 or containers["stdout"]:
            raise RuntimeError("Container check failed or running containers appeared; preparation stopped")
    elif probe(["systemctl", "is-active", "docker"])["stdout"] == "active":
        raise RuntimeError("Active Docker daemon cannot be inspected; preparation stopped")


def prepare(actions):
    logger.info("Starting host preparation with actions: %s", actions)
    allowed = {"install-docker", "start-docker", "install-toolkit",
               "configure-runtime", "verify-gpu"}
    if not actions or any(action not in allowed for action in actions):
        raise ValueError("Invalid preparation actions")
    if os.geteuid() != 0:
        raise PermissionError("Preparation requires root")
    release = os_release()
    if (release.get("ID") != "ubuntu" or release.get("VERSION_ID") not in ("22.04", "24.04")
            or os.uname().machine != "x86_64"):
        raise RuntimeError("Unsupported preparation host")
    for action in actions:
        if action != "verify-gpu":
            ensure_idle()
        if action == "install-docker":
            run(["apt-get", "update"])
            run(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "docker.io"])
            run(["systemctl", "start", "docker"])
        elif action == "start-docker":
            run(["systemctl", "start", "docker"])
        elif action == "install-toolkit":
            run(["apt-get", "update"])
            run(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y",
                 "--no-install-recommends", "ca-certificates", "curl", "gnupg"])
            with tempfile.TemporaryDirectory(prefix="inference-platform-") as directory:
                key = Path(directory) / "key"
                listing = Path(directory) / "repository.list"
                run(["curl", "-fsSL", "https://nvidia.github.io/libnvidia-container/gpgkey",
                     "-o", str(key)])
                run(["gpg", "--batch", "--yes", "--dearmor", "-o",
                     "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg", str(key)])
                run(["curl", "-fsSL",
                     "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list",
                     "-o", str(listing)])
                source = listing.read_text().replace(
                    "deb https://",
                    "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://")
                Path("/etc/apt/sources.list.d/nvidia-container-toolkit.list").write_text(source)
            run(["apt-get", "update"])
            run(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y",
                 "nvidia-container-toolkit"])
        elif action == "configure-runtime":
            config = Path("/etc/docker/daemon.json")
            if config.exists():
                # Unique backup avoids overwriting the original on retries.
                with tempfile.NamedTemporaryFile(prefix="daemon.json.platform-",
                                                 dir=str(config.parent), delete=False) as backup:
                    backup.write(config.read_bytes())
                    print("Docker configuration backup: " + backup.name, flush=True)
            run(["nvidia-ctk", "runtime", "configure", "--runtime=docker"])
            run(["systemctl", "restart", "docker"])
        elif action == "verify-gpu":
            run(DOCKER + ["run", "--rm", "--network", "none", "--gpus", "all",
                          CUDA_IMAGE, "nvidia-smi", "-L"])


if __name__ == "__main__":
    configure_logging()
    if sys.argv[1:] == ["inspect"]:
        print(json.dumps(inspect()))
    elif len(sys.argv) > 2 and sys.argv[1] == "prepare":
        prepare(sys.argv[2:])
    else:
        raise SystemExit("Expected inspect or prepare <actions>")
