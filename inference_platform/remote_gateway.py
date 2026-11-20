"""Install the generated site on a systemd NGINX host."""

import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile


SITE = Path("/etc/nginx/conf.d/inference-platform.conf")
TLS_DIR = Path("/etc/nginx/inference-platform")
MANAGED = b"# Managed by inference-platform gateway deploy\n"
logger = logging.getLogger("inference_platform.remote_gateway")


def run_checked(command):
    logger.debug("Running %s", " ".join(command))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{' '.join(command)} timed out") from error
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr.strip()}")
    return result


def deploy_gateway(config):
    _with_gateway_lock(_install_site, MANAGED + config)


def _with_gateway_lock(operation, payload):
    directory = os.open(SITE.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        operation(payload)
    except BlockingIOError as error:
        raise RuntimeError("Another gateway deployment is in progress") from error
    finally:
        os.close(directory)


def bootstrap_tls(payload):
    name = payload.get("server_name") if isinstance(payload, dict) else None
    token = payload.get("token") if isinstance(payload, dict) else None
    if (not isinstance(name, str) or len(name) > 253 or "." not in name
            or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                       for label in name.split("."))):
        raise ValueError("Expected a fully qualified DNS name for the lab certificate")
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid bearer token format")
    _with_gateway_lock(_install_lab_credentials, (name, token))


def _install_lab_credentials(credentials):
    name, token = credentials
    if os.path.lexists(TLS_DIR):
        raise RuntimeError(f"Refusing to overwrite existing credential directory {TLS_DIR}")
    staging = Path(tempfile.mkdtemp(prefix=".inference-platform-tls-", dir=TLS_DIR.parent))
    try:
        logger.info("Generating a 30-day self-signed lab certificate for %s", name)
        key = staging / "privkey.pem"
        cert = staging / "fullchain.pem"
        run_checked([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
            "-days", "30", "-subj", "/CN=inference-platform",
            "-addext", f"subjectAltName=DNS:{name}", "-addext", "basicConstraints=critical,CA:FALSE",
            "-addext", "extendedKeyUsage=serverAuth", "-keyout", str(key), "-out", str(cert),
        ])
        key.chmod(0o600)
        token_map = staging / "bearer-token.map"
        token_map.write_text(f'"~^Bearer[ ]{token}$" 1;\n', encoding="ascii")
        token_map.chmod(0o600)
        staging.rename(TLS_DIR)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _install_site(config):
    try:
        existing = SITE.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        not stat.S_ISREG(existing.st_mode) or not SITE.read_bytes().startswith(MANAGED)
    ):
        raise RuntimeError(f"Refusing to overwrite unmanaged site {SITE}")

    logger.info("Checking that the NGINX service is active")
    run_checked(["systemctl", "is-active", "--quiet", "nginx"])
    staging = Path(tempfile.mkdtemp(prefix=".inference-platform-", dir=SITE.parent))
    backup = staging / "previous"
    keep_backup = False
    try:
        candidate = staging / "candidate"
        candidate.write_bytes(config)
        candidate.chmod(0o644)
        if existing is not None:
            os.link(SITE, backup)
        logger.info("Installing candidate site at %s", SITE)
        os.replace(candidate, SITE)
        applied = False
        reload_started = False
        try:
            # -T validates the full config and confirms the service config includes this site.
            logger.info("Validating NGINX configuration and checking that it includes the managed site")
            validation = run_checked(["nginx", "-T"])
            if "conflicting server name" in validation.stderr:
                raise RuntimeError(f"Resolve conflicting NGINX sites first:\n{validation.stderr.strip()}")
            if f"# configuration file {SITE}:" not in validation.stdout.splitlines():
                raise RuntimeError(f"NGINX does not include {SITE}")
            reload_started = True
            logger.info("Validation passed; requesting NGINX reload")
            run_checked(["systemctl", "reload", "nginx"])
            applied = True
            logger.info("NGINX reload command succeeded")
        finally:
            if not applied:
                try:
                    if existing is None:
                        SITE.unlink()
                    else:
                        os.replace(backup, SITE)
                except OSError as error:
                    keep_backup = True
                    raise RuntimeError(
                        f"Could not restore {SITE}; inspect {staging} before retrying: {error}"
                    ) from error
                if reload_started:
                    logger.warning("Site rolled back on disk. Check running NGINX before retrying.")
                else:
                    logger.info("Site rolled back on disk without reloading NGINX")
    finally:
        if not keep_backup:
            shutil.rmtree(staging)


if __name__ == "__main__":
    try:
        configure_logging()
        if os.geteuid() != 0:
            raise RuntimeError("Gateway deploy requires administrative access")
        operation = sys.argv[1]
        if operation == "deploy":
            deploy_gateway(sys.stdin.buffer.read())
        elif operation == "bootstrap-tls":
            bootstrap_tls(json.load(sys.stdin))
        else:
            raise ValueError("Expected deploy or bootstrap-tls")
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
