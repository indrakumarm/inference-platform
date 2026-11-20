import argparse
import getpass
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import sys
import warnings

from .logging_config import configure_logging
from .gateway_check import check_gateway
from .ssh import SSHHost


logger = logging.getLogger(__name__)
_FIELDS = {
    "listen", "backends", "max_in_flight", "connect_timeout_seconds",
    "upstream_read_timeout_seconds",
}
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.ASCII | re.IGNORECASE)
_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(network) for network in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
))
_TLS_DIR = "/etc/nginx/inference-platform"
_TOKEN_MAP = f"    include {_TLS_DIR}/bearer-token.map;"


def _dns_name(host):
    return (isinstance(host, str) and len(host) <= 253
            and all(_DNS_LABEL.fullmatch(label) for label in host.split(".")))


def _integer(value, field, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def _endpoint(value, field, *, listener=False, secure=False):
    if not isinstance(value, dict) or set(value) != {"host", "port"}:
        raise ValueError(f"{field} must contain exactly host and port")
    host = value["host"]
    if listener and secure and type(value["port"]) is int and value["port"] == 443:
        port = 443
    else:
        port = _integer(value["port"], f"{field}.port", 1024 if listener else 1, 65535)
    if listener:
        allowed = host == "127.0.0.1"
        if secure and isinstance(host, str):
            try:
                address = ipaddress.IPv4Address(host)
            except ipaddress.AddressValueError:
                pass
            else:
                allowed |= any(address in network for network in _PRIVATE_NETWORKS)
        if not allowed:
            raise ValueError("listen.host must be 127.0.0.1 or, with TLS, a private RFC 1918 IPv4 address")
    else:
        if not isinstance(host, str) or "%" in host:
            raise ValueError(f"{field}.host must be an IP address or DNS hostname")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if host.lower() == "unix" or not _dns_name(host):
                raise ValueError(f"{field}.host must be an IP address or DNS hostname") from None
            host = host.lower()
        else:
            host = f"[{address}]" if address.version == 6 else str(address)
    return f"{host}:{port}"


def render_gateway(profile):
    if not isinstance(profile, dict) or set(profile) - {"tls"} != _FIELDS:
        raise ValueError("Gateway profile requires " + ", ".join(sorted(_FIELDS)) + "; tls is optional")
    secure = "tls" in profile
    server_name = "localhost"
    tls_settings = auth_map = auth_check = listen_option = ""
    if secure:
        tls = profile["tls"]
        if (not isinstance(tls, dict) or set(tls) != {"server_name"}
                or not _dns_name(tls["server_name"]) or "." not in tls["server_name"]):
            raise ValueError("tls must contain only server_name, a fully qualified DNS name")
        server_name = tls["server_name"].lower()
        listen_option = " ssl"
        tls_settings = (
            f"\n    ssl_certificate {_TLS_DIR}/fullchain.pem;\n"
            f"    ssl_certificate_key {_TLS_DIR}/privkey.pem;\n"
            "    ssl_protocols TLSv1.2 TLSv1.3;\n"
        )
        auth_map = (
            "map $http_authorization $inference_bearer_allowed {\n"
            f"    default 0;\n{_TOKEN_MAP}\n}}\n\n"
        )
        auth_check = """        if ($inference_bearer_allowed = 0) {
            add_header WWW-Authenticate 'Bearer realm="inference"' always;
            return 401;
        }
        proxy_set_header Authorization "";

"""
    listen = _endpoint(profile["listen"], "listen", listener=True, secure=secure)
    backends = profile["backends"]
    if not isinstance(backends, list) or not backends:
        raise ValueError("backends must be a non-empty list")
    endpoints = [_endpoint(backend, f"backends[{index}]") for index, backend in enumerate(backends)]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("backends must not contain duplicate endpoints")
    if listen in endpoints:
        raise ValueError("a backend must not point at the gateway listener")
    servers = "\n".join(f"    server {endpoint};" for endpoint in endpoints)
    limit = _integer(profile["max_in_flight"], "max_in_flight", 1, 65535)
    connect = _integer(profile["connect_timeout_seconds"], "connect_timeout_seconds", 1, 75)
    read = _integer(profile["upstream_read_timeout_seconds"], "upstream_read_timeout_seconds", 1, 86400)
    return f"""upstream inference_vllm {{
    zone inference_vllm 1m;
    least_conn;
{servers}
}}

{auth_map}limit_conn_zone $server_name zone=inference_inflight:1m;

log_format inference_timing '$time_iso8601 id=$request_id status=$status '
                            'upstream=$upstream_addr upstream_status=$upstream_status '
                            'request_time=$request_time '
                            'upstream_connect_time=$upstream_connect_time '
                            'upstream_header_time=$upstream_header_time '
                            'upstream_response_time=$upstream_response_time';

server {{
    listen {listen}{listen_option};
    server_name {server_name};
{tls_settings}
    location = /v1/chat/completions {{
        access_log /var/log/nginx/inference_access.log inference_timing;
{auth_check}        limit_conn inference_inflight {limit};
        limit_conn_status 429;

        proxy_pass http://inference_vllm;
        proxy_http_version 1.1;
        proxy_buffering off;
        proxy_connect_timeout {connect}s;
        proxy_read_timeout {read}s;
        proxy_next_upstream off;
    }}

    location / {{
        return 404;
    }}
}}
"""


def gateway_main(argv):
    parser = argparse.ArgumentParser(
        prog="inference_platform gateway", description="Configure an NGINX gateway.")
    commands = parser.add_subparsers(dest="operation", required=True)
    render = commands.add_parser("render", help="Print NGINX config without contacting a host")
    deploy = commands.add_parser("deploy", help="Install, validate and reload NGINX over SSH")
    bootstrap = commands.add_parser(
        "bootstrap-tls", help="Provision a self-signed lab certificate and bearer token over SSH")
    check = commands.add_parser("check", help="Verify HTTPS, authentication and streaming from this client")
    check.add_argument("--ca-file", type=Path, help="Trusted CA or public lab certificate (default: system trust)")
    check.add_argument("--model", required=True, help="Client-visible vLLM model ID")
    check.add_argument("--timeout", type=int, default=30, help="Total check deadline in seconds (default: 30)")
    for command in (render, deploy, bootstrap, check):
        command.add_argument("--profile", required=True, type=Path, help="Gateway JSON profile")
    for command in (deploy, bootstrap):
        command.add_argument("--host", required=True, help="NGINX host (user@hostname or user@IPv4)")
        command.add_argument("--key", help="SSH private key path; otherwise use SSH agent/config")
        command.add_argument("--port", type=int, default=22, help="SSH port")
    args = parser.parse_args(argv)
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        config = render_gateway(profile)
        if args.operation in ("bootstrap-tls", "check") and "tls" not in profile:
            raise ValueError(f"{args.operation} requires an HTTPS gateway profile with tls.server_name")
    except json.JSONDecodeError as error:
        parser.error(f"{args.profile}: expected a JSON gateway profile ({error})")
    except (OSError, ValueError) as error:
        parser.error(f"{args.profile}: {error}")
    if args.operation == "render":
        print(config, end="")
        return 0
    if args.operation == "check":
        if not 1 <= args.timeout <= 300:
            check.error("--timeout must be between 1 and 300 seconds")
        token = os.environ.get("API_TOKEN", "")
        if not re.fullmatch(r"[0-9a-f]{64}", token):
            check.error("set API_TOKEN to the gateway's 64-character lowercase hex bearer token")
        try:
            ttft, total = check_gateway(
                profile["tls"]["server_name"].lower(), profile["listen"]["port"],
                args.ca_file, args.model, token, args.timeout,
            )
        except (OSError, RuntimeError) as error:
            print(f"Gateway check failed: {error}", file=sys.stderr)
            return 1
        print(f"PASS: authenticated stream reached [DONE] (TTFT {ttft:.3f}s, total {total:.3f}s).")
        return 0

    if not sys.stdin.isatty():
        print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
        return 2
    try:
        configure_logging()
        host = SSHHost(args.host, args.key, args.port)
        if args.operation == "bootstrap-tls":
            name = profile["tls"]["server_name"].lower()
            print(f"Provision a 30-day self-signed certificate for {name} on {args.host}.")
            if input("Install lab TLS credentials without reloading NGINX? [y/N]: ").strip().lower() not in ("y", "yes"):
                print("Cancelled; nothing changed.")
                return 0
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    token = getpass.getpass("Bearer token (64 lowercase hex characters): ")
                except getpass.GetPassWarning as error:
                    raise RuntimeError("Cannot read the token without terminal echo") from error
            if not re.fullmatch(r"[0-9a-f]{64}", token):
                raise ValueError("Bearer token must contain exactly 64 lowercase hex characters")
            host.gateway_bootstrap_tls(name, token)
            print(f"Lab TLS credentials installed on {args.host}. Run gateway deploy to activate.")
            return 0
        print(f"Proposed NGINX config for {args.host} (preview only; host not contacted):")
        print(config, end="")
        answer = input(f"Deploy config and reload NGINX on {args.host}? [y/N]: ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled; nothing changed.")
            return 0
        host.gateway_deploy(config)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted; if deployment began, check the remote NGINX state.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Gateway %s failed: %s", args.operation, error)
        return 1
    print(f"NGINX config validated and reload requested successfully on {args.host}.")
    return 0
