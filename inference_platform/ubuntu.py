from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class Plan:
    blockers: list[str]
    actions: list[str]


ACTION_DESCRIPTIONS = {
    "install-docker": "Install Ubuntu docker.io and start the rootful Docker service.",
    "start-docker": "Start the existing rootful Docker service.",
    "install-toolkit": "Add NVIDIA's signed stable apt repository and install NVIDIA Container Toolkit.",
    "configure-runtime": "Back up daemon.json, register NVIDIA runtime and restart Docker.",
    "verify-gpu": "Pull/run nvidia/cuda:12.4.1-base-ubuntu22.04 to verify Docker GPU access.",
}


def plan(report):
    """Plan supported host changes, never infer readiness from missing probe data."""
    blockers = []
    actions = []
    os_info = report["os"]
    checks = report["checks"]
    if os_info.get("ID") != "ubuntu" or os_info.get("VERSION_ID") not in ("22.04", "24.04"):
        blockers.append("Automatic preparation supports Ubuntu 22.04/24.04 only.")
    if report["architecture"] != "x86_64":
        blockers.append("Automatic preparation currently supports x86_64 hosts only.")
    if not report["privileged"]:
        blockers.append("Preparation requires root or non-interactive sudo; inspect remains available.")
    if checks["gpu"]["code"] != 0 or not checks["gpu"]["stdout"].strip():
        blockers.append("nvidia-smi failed or returned no GPUs. Fix the driver/GPU first; no driver installation is automated.")
    if checks["gpu_processes"]["code"] != 0:
        blockers.append("Cannot verify GPU workloads; investigate nvidia-smi process query before preparation.")
    elif checks["gpu_processes"]["stdout"].strip():
        blockers.append("GPU compute processes are active. Stop only your own workloads before preparation.")
    if report["root_free_bytes"] < 5 * 1024**3:
        blockers.append("Less than 5 GiB is free on /. Free space before installing packages.")
    if checks["systemd"]["code"] != 0:
        blockers.append("A systemd-managed host is required; nested/container hosts are not prepared automatically.")
    docker_ready = checks["docker_info"]["code"] == 0
    if not report["docker_installed"]:
        actions.append("install-docker")
    elif not docker_ready:
        if checks["docker_active"]["stdout"].strip() == "active":
            blockers.append("Docker service is active but cannot be queried. Fix access/daemon errors manually.")
        else:
            actions.append("start-docker")
    if not report["toolkit_installed"]:
        actions.append("install-toolkit")
    if not docker_ready or "nvidia" not in checks["docker_info"]["stdout"].split():
        actions.append("configure-runtime")
    # Docker/runtime changes are host-wide, even when only one model is intended.
    if docker_ready and actions:
        if checks["containers"]["code"] != 0:
            blockers.append("Cannot enumerate running containers; refusing host-wide changes.")
        elif checks["containers"]["stdout"].strip():
            blockers.append("Containers are running. Preparation will not restart/change Docker under active workloads.")
    actions.append("verify-gpu")
    logger.debug("Host preparation plan: %d blocker(s), actions=%s", len(blockers), actions)
    return Plan(blockers, actions)
