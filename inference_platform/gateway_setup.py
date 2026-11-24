import getpass
import logging
import os
from pathlib import Path
import re
import ssl
import sys
import warnings

from .deployment import load_profile
from .gateway import _dns_name, _endpoint, render_gateway
from .gateway_check import check_gateway, check_http_gateway
from .logging_config import configure_logging
from .platform_config import load_api_token, load_platform_config
from .providers import load_target
from .ssh import SSHHost


logger = logging.getLogger(__name__)


def _yes(prompt):
    return input(prompt).strip().lower() in ("y", "yes")


def _token(env_path=None, required_file=False):
    file_token = load_api_token(env_path, required_file=required_file) if env_path else None
    token = os.environ.get("API_TOKEN", file_token)
    if token is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            raise ValueError("API_TOKEN must contain exactly 64 lowercase hex characters")
        return token
    if not sys.stdin.isatty():
        raise RuntimeError("Set API_TOKEN in the environment or .env for non-interactive verification")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        try:
            token = getpass.getpass("Bearer token (64 lowercase hex characters): ")
        except getpass.GetPassWarning as error:
            raise RuntimeError("Cannot read the token without terminal echo") from error
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Bearer token must contain exactly 64 lowercase hex characters")
    return token


def _port(prompt, default):
    value = input(prompt).strip()
    try:
        return int(value) if value else default
    except ValueError as error:
        raise ValueError("Gateway port must be an integer") from error


def _client_check(profile, model, token=None, *, settings=None, target=None,
                  env_path=None, required_env=False):
    mode = "https" if "tls" in profile else "http"
    client_port = (settings.get("client_port", profile["listen"]["port"])
                   if settings is not None else profile["listen"]["port"])
    _endpoint({"host": "127.0.0.1", "port": client_port}, "client", listener=True, secure=mode == "https")
    if mode == "https":
        print("The DNS name must resolve to the gateway from this client. "
              "For a lab certificate, obtain the public certificate over trusted SSH.")
        if token is None:
            token = _token(env_path, required_env)
        if settings is None or "ca_file" not in settings:
            ca = _ask("CA/public certificate path (empty for system trust): ")
            ca = Path(ca).expanduser() if ca else None
        else:
            ca = settings.get("ca_file")
            if settings.get("fetch_lab_certificate") and not ca.exists():
                host = SSHHost(target.host, target.key, target.port, target.known_hosts)
                certificate = host.gateway_certificate()
                try:
                    ssl.PEM_cert_to_DER_cert(certificate)
                except ValueError as error:
                    raise ValueError("Gateway did not return a valid public PEM certificate") from error
                fd = os.open(ca, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "w") as stream:
                        stream.write(certificate)
                except OSError:
                    ca.unlink(missing_ok=True)
                    raise
                print(f"Saved public certificate from the trusted SSH host to {ca}.")
        first, total = check_gateway(profile["tls"]["server_name"], client_port, ca, model, token, 30)
        print("PASS: HTTPS authentication and streaming verified from this client.")
    else:
        first, total = check_http_gateway("127.0.0.1", client_port, model, 30)
        print("PASS: HTTP streaming verified from this client; HTTP has no authentication.")
    print(f"PASS: received generated text and [DONE] (TTFT {first:.3f}s, total {total:.3f}s).")


def _configuration(target_path, deployment_path, config_path, env_file):
    if config_path:
        config = load_platform_config(config_path)
        return config.target, config.deployment, config.gateway, env_file or config.env_file
    return load_target(target_path), load_profile(deployment_path), None, env_file or Path(".env")


def _ask(prompt, default=None):
    if not sys.stdin.isatty():
        raise RuntimeError(f"Missing required configuration: {prompt}")
    value = input(prompt).strip()
    return value or default


def _profile(deployment, settings, *, verification=False):
    gateway = settings or {}
    mode = gateway.get("mode") or _ask("Gateway mode: local HTTP or HTTPS? [http/https]: ")
    if not isinstance(mode, str) or mode.lower() not in ("http", "https"):
        raise ValueError("Choose http or https")
    mode = mode.lower()
    listen = gateway.get("listen", {})
    if mode == "http" and (gateway.get("tls") or gateway.get("ca_file")
                            or gateway.get("fetch_lab_certificate")):
        raise ValueError("HTTP gateway cannot have TLS settings")
    host = listen.get("host")
    if host is None:
        if mode == "https" and not verification:
            host = _ask("Gateway listen IPv4 [127.0.0.1]: ", "127.0.0.1")
        else:
            host = "127.0.0.1"
    default_port = 8080 if mode == "http" else 8443
    port = listen.get("port")
    if port is None:
        if not sys.stdin.isatty():
            raise RuntimeError("Missing required configuration: gateway.listen.port")
        port = _port(("Local tunnel port [8080]: " if verification else
                      "Gateway loopback port [8080]: ") if mode == "http"
                     else "Gateway HTTPS port [8443]: ", default_port)
    profile = {
        "listen": {"host": host, "port": port},
        "backends": [{"host": "127.0.0.1", "port": deployment.port}],
        "max_in_flight": 1, "connect_timeout_seconds": 5,
        "upstream_read_timeout_seconds": 120,
    }
    if mode == "https":
        name = gateway.get("tls", {}).get("server_name")
        if name is None:
            name = _ask("Gateway DNS name: ")
        if not isinstance(name, str) or not _dns_name(name) or "." not in name:
            raise ValueError("Gateway DNS name must be a fully qualified DNS name")
        profile["tls"] = {"server_name": name.lower()}
    if "client_port" in gateway:
        _endpoint({"host": "127.0.0.1", "port": gateway["client_port"]},
                  "client", listener=True, secure=mode == "https")
    if verification:
        _endpoint(profile["listen"], "listen", listener=True, secure=mode == "https")
    else:
        render_gateway(profile)
    return profile


def verify_gateway(target_path, deployment_path, *, config_path=None, env_file=None):
    if not sys.stdin.isatty() and not config_path:
        print("Interactive verification requires a terminal.", file=sys.stderr)
        return 2
    try:
        configure_logging()
        target, deployment, settings, env_path = _configuration(
            target_path, deployment_path, config_path, env_file)
        if target.provider != "ssh":
            raise ValueError("Gateway verification currently requires an ssh target")
        profile = _profile(deployment, settings, verification=True)
        if "tls" not in profile:
            print(f"Connect an SSH tunnel to the gateway loopback on {target.host} before verifying.")
        _client_check(profile, deployment.model, settings=settings, target=target,
                      env_path=env_path, required_env=env_file is not None)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nVerification interrupted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("Gateway verification failed: %s", error)
        return 1


def setup_gateway(target_path, deployment_path, *, config_path=None, env_file=None):
    if not sys.stdin.isatty():
        print("Interactive setup requires a terminal; nothing changed.", file=sys.stderr)
        return 2
    credentials_created = False
    credentials_attempted = False
    deploy_attempted = False
    deployed = False
    try:
        configure_logging()
        target, deployment, settings, env_path = _configuration(
            target_path, deployment_path, config_path, env_file)
        if target.provider != "ssh":
            raise ValueError("Gateway setup currently requires an ssh target")
        host = SSHHost(target.host, target.key, target.port, target.known_hosts)
        profile = _profile(deployment, settings)
        mode = "https" if "tls" in profile else "http"
        config = render_gateway(profile)
        print(f"Checking NGINX and model {deployment.model} on {target.host}...")
        facts = host.gateway_inspect(deployment.port, deployment.model)
        print("PASS: NGINX is active; backend reports the expected model.")

        create_credentials = False
        if mode == "https":
            if facts["credentials_exist"]:
                print("Existing TLS credentials found; they will not be overwritten. "
                      "They must match the chosen DNS name and token.")
                if not _yes("Use the existing credentials? [y/N]: "):
                    print("Cancelled; nothing changed.")
                    return 0
            else:
                print("No TLS credentials found. Setup can create a 30-day self-signed lab certificate.")
                if not _yes("Create lab TLS credentials? [y/N]: "):
                    print("Cancelled; nothing changed.")
                    return 0
                create_credentials = True
        print(f"\nProposed NGINX configuration for {target.host}:")
        print(config, end="")
        if mode == "https":
            print("Restrict network access separately; no firewall or DNS changes will be made.")
        else:
            print("HTTP is unauthenticated and replaces any existing managed gateway configuration. "
                  "Only trusted users of the gateway host can access its loopback listener.")
        if not _yes("Apply this configuration and reload NGINX? [y/N]: "):
            print("Cancelled; nothing changed.")
            return 0
        token = None
        if create_credentials:
            print("Save the token in your secrets manager; it is not saved locally by setup.")
            token = _token(env_path, env_file is not None)
            credentials_attempted = True
            host.gateway_bootstrap_tls(profile["tls"]["server_name"], token)
            credentials_created = True
            print("Lab TLS credentials created; NGINX configuration is not yet deployed.")
        deploy_attempted = True
        host.gateway_deploy(config)
        deployed = True
        print("PASS: NGINX configuration validated and reload requested.")
        if mode == "http" or profile["listen"]["host"] == "127.0.0.1":
            print(f"The gateway listens on the remote host's loopback port {profile['listen']['port']}. "
                  "Before client verification, establish an SSH tunnel to that port.")
        if not _yes("Verify the gateway from this client now? [y/N]: "):
            if config_path:
                print(f"Gateway deployed; client verification skipped. Run gateway verify --config {config_path}.")
            else:
                print("Gateway deployed; client verification skipped. "
                      f"Run gateway verify with --target {target_path} "
                      f"--deployment-profile {deployment_path} when ready.")
            return 0
        _client_check(profile, deployment.model, token, settings=settings, target=target,
                      env_path=env_path, required_env=env_file is not None)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted; check gateway state before retrying.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("Gateway setup failed: %s", error)
        if deployed:
            print("NGINX was reloaded, but client verification failed. Check DNS, reachability, "
                  "certificate trust and credentials before retrying.", file=sys.stderr)
        elif credentials_created:
            print("Lab credentials were created but deployment did not complete. "
                  "Rerun setup and choose existing credentials after checking the host.", file=sys.stderr)
            if deploy_attempted:
                print("The reload may have been attempted; check the running NGINX service.", file=sys.stderr)
        elif credentials_attempted:
            print("Credential creation may have completed; inspect the gateway host before retrying.",
                  file=sys.stderr)
        elif deploy_attempted:
            print("Deployment may have changed the gateway; check the running NGINX service before retrying.",
                  file=sys.stderr)
        return 1
