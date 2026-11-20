"""Update the platform-owned Prometheus file discovery directory."""

import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import tempfile


TARGET_DIR = Path("/etc/prometheus/targets/inference-platform")


def update_target(action, payload):
    ip = ipaddress.IPv4Address(payload["ip"])
    port = payload["port"]
    name = payload["name"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("Invalid scrape port")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Invalid deployment name")

    address = f"{ip}:{port}"
    identifier = hashlib.sha256(f"{name}/{address}".encode()).hexdigest()
    path = TARGET_DIR / f"vllm-{identifier}.json"
    if action == "remove":
        if not TARGET_DIR.is_dir():
            raise RuntimeError(f"Missing target directory: {TARGET_DIR}")
        path.unlink(missing_ok=True)
    elif action == "add":
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=TARGET_DIR,
                                         prefix=".vllm-", delete=False) as candidate:
            temporary = Path(candidate.name)
            try:
                json.dump([{"targets": [address], "labels": {"deployment": name}}], candidate)
                candidate.write("\n")
                candidate.close()
                temporary.chmod(0o644)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    else:
        raise ValueError("Unsupported target update")


if __name__ == "__main__":
    try:
        update_target(sys.argv[1], json.load(sys.stdin))
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
