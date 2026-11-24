import argparse
import json
import logging
import subprocess
import sys
from .deployment import load_profile
from .gateway import gateway_main
from .logging_config import configure_logging
from .providers import DEFAULT_PROVIDER, PROVIDERS, Target, create_provider, load_target
from .ssh import SSHHost
from .ubuntu import ACTION_DESCRIPTIONS


logger = logging.getLogger(__name__)


def show_deployment(profile, blockers):
    print("\nProposed deployment:")
    print(f"  Name: {profile.name}")
    print(f"  Image: {profile.image}")
    print(f"  Model: {profile.model}@{profile.model_revision}")
    print(f"  Tokenizer: {profile.tokenizer}@{profile.tokenizer_revision}")
    print(f"  GPUs: {', '.join(map(str, profile.gpu_indices))}; "
          f"context: {profile.max_model_len}; dtype: {profile.dtype}")
    print(f"  Local endpoint: http://127.0.0.1:{profile.port}/v1")
    if profile.private_bind_ip:
        print(f"  Private endpoint: http://{profile.private_bind_ip}:{profile.port}")
        print("  WARNING: The private binding exposes the entire unauthenticated vLLM API. "
              "Restrict the port to trusted hosts before deploying.")
    for blocker in blockers:
        print("BLOCKED: " + blocker)


def show_report(report, proposal):
    print(f"\nHost: {report['os'].get('PRETTY_NAME', 'unknown OS')} / {report['architecture']}")
    print(f"Free disk: / {report['root_free_bytes'] / 1024**3:.1f} GiB; "
          f"login home {report['cache_free_bytes'] / 1024**3:.1f} GiB")
    if report.get("runtime") == "vast-container":
        print("Runtime: existing Vast container (no package installation)")
    else:
        print("Non-interactive administrative access: " + ("yes" if report["privileged"] else "no"))
    for name, result in report["checks"].items():
        detail = result["stdout"] or result["stderr"] or "(no output)"
        print(f"  {name} [exit {result['code']}]: {detail}")
    print("\nProposed actions (not yet executed):")
    for action in proposal.actions:
        print("  - " + ACTION_DESCRIPTIONS[action])
    for blocker in proposal.blockers:
        print("BLOCKED: " + blocker)
    print("This is host-runtime readiness, not proof that a particular model fits or runs.")


def show_container_status(status, kind="container"):
    print(f"\n{kind.capitalize()}: {status['name']}")
    print(f"State: {status['state']}")
    if not status["exists"]:
        return
    print(f"Running: {'yes' if status['running'] else 'no'}")
    if status["running"]:
        print(f"Started: {status['started_at']}")
    else:
        exit_code = status["exit_code"]
        print(f"Last exit code: {exit_code if exit_code is not None else 'unknown'}")
        if status["error"]:
            print("Error: " + status["error"])
        if status["finished_at"]:
            print(f"Finished: {status['finished_at']}")


def show_deployments(deployments):
    if not deployments:
        print("No inference-platform deployments found on this host.")
        return
    print("\nDeployments:")
    for deployment in deployments:
        model = deployment["model"]
        ports = deployment["ports"] or "no published ports"
        print(f"  {deployment['container']} [{deployment['state']}] "
              f"profile={deployment['profile']} model={model} ports={ports}")
        if deployment["metadata_source"] == "container-name":
            print("    Legacy container: model/profile metadata is not available in Docker labels.")


def update_monitoring_target(monitor, action, profile):
    try:
        monitor.update_monitoring_target(action, profile.private_bind_ip, profile.port, profile.name)
    except (OSError, RuntimeError) as error:
        state = "Model deployed" if action == "add" else "Container absent"
        print(f"{state}, but Prometheus target update failed: {error}", file=sys.stderr)
        if action == "remove":
            print("Retry remove with the same profile and monitoring host to clean up the target.",
                  file=sys.stderr)
        else:
            print("The model is still running; repair the target file on the monitoring host.",
                  file=sys.stderr)
        return 1
    print(f"Prometheus target {'registered' if action == 'add' else 'removed'}: "
          f"{profile.private_bind_ip}:{profile.port}")
    return 0


def run_lifecycle(host, operation, target, profile, monitor=None, kind="container"):
    """Drive a lifecycle operation; ``host`` is a provider and ``target`` its display name."""
    status = host.lifecycle("status", profile.as_dict())
    show_container_status(status, kind)
    if operation == "status":
        if status["state"] in ("unknown", "orphaned"):
            print("Process ownership is uncertain; inspect the host before taking action.", file=sys.stderr)
            return 1
        return 0
    if status["state"] in ("unknown", "orphaned", "starting"):
        print(f"Cannot safely manage {kind} in state {status['state']}; inspect the host.",
              file=sys.stderr)
        return 2
    cleanup_only = not status["exists"] and operation == "remove" and monitor is not None
    if not status["exists"] and not cleanup_only:
        print(f"No {kind} exists for this profile; deploy it first.", file=sys.stderr)
        return 2
    if operation == "start" and status["running"]:
        print(f"{kind.capitalize()} is already running.")
        return 0
    if operation == "stop" and not status["running"]:
        print(f"{kind.capitalize()} is already stopped.")
        return 0
    if operation == "remove" and status["running"]:
        print(f"Stop the {kind} before removing it.", file=sys.stderr)
        return 2

    if not cleanup_only:
        print(f"\nThis action affects only the named inference-platform {kind}.")
    if kind == "process" and operation == "start":
        print("The running container image digest cannot be independently verified here.")
    if operation == "remove" and not cleanup_only:
        if kind == "process":
            print("Removing the process record keeps the model cache and logs.")
        else:
            print("Removing the container keeps its image and Hugging Face cache volume.")
    if monitor is not None:
        print(f"Will remove the Prometheus target {profile.private_bind_ip}:{profile.port} on {monitor.target}.")
    if not sys.stdin.isatty():
        print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
        return 2
    answer = input(f"{operation.capitalize()} {status['name']} on {target}? [y/N]: ")
    if answer.strip().lower() not in ("y", "yes"):
        print("Cancelled; nothing changed.")
        return 0

    latest = host.lifecycle("status", profile.as_dict())
    if (latest["exists"] != status["exists"] or latest["state"] != status["state"]
            or latest["running"] != status["running"]):
        show_container_status(latest, kind)
        print(f"{kind.capitalize()} state changed; review it and rerun the command.", file=sys.stderr)
        return 2
    if cleanup_only:
        return update_monitoring_target(monitor, "remove", profile)
    result = host.lifecycle(operation, profile.as_dict())
    if operation == "remove":
        if kind == "process":
            print(f"Removed {result['name']}. The model cache and logs were retained.")
        else:
            print(f"Removed {result['name']}. The model cache volume was retained.")
        if monitor is not None:
            return update_monitoring_target(monitor, "remove", profile)
    else:
        show_container_status(result, kind)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "gateway":
        return gateway_main(argv[1:])
    if argv and argv[0] == "infra":
        from .infra.cli import infra_main
        return infra_main(argv[1:])
    parser = argparse.ArgumentParser(
        description="Inspect, prepare, deploy and manage a GPU host.",
        epilog="Other commands: inference_platform gateway --help; inference_platform infra --help")
    lifecycle_operations = ("status", "start", "stop", "remove")
    profile_operations = ("deploy", *lifecycle_operations)
    parser.add_argument("operation", choices=("inspect", "prepare", "list", *profile_operations))
    parser.add_argument("--host", help="user@hostname or user@IPv4")
    parser.add_argument("--key", help="SSH private key path; otherwise use SSH agent/config")
    parser.add_argument("--port", type=int, help="SSH port (default 22)")
    parser.add_argument("--known-hosts", help="known_hosts file for this host; otherwise SSH defaults")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        help=f"Host provider adapter (default {DEFAULT_PROVIDER})")
    parser.add_argument("--target", help="JSON target file replacing --provider/--host/--port/--key/--known-hosts")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable output; supported by inspect and list")
    parser.add_argument("--profile", help="JSON deployment profile; required for deploy/lifecycle")
    parser.add_argument("--monitoring-host", help="SSH target for Prometheus file discovery (deploy/remove)")
    parser.add_argument("--monitoring-key", help="Monitoring SSH key; otherwise use SSH agent/config")
    args = parser.parse_args(argv)
    connection_flags = (args.host, args.key, args.port, args.known_hosts, args.provider)
    if args.target and any(flag is not None for flag in connection_flags):
        parser.error("--target cannot be combined with --provider, --host, --port, --key or --known-hosts")
    if not args.target and not args.host:
        parser.error("--host or --target is required")
    if args.json and args.operation not in ("inspect", "list"):
        parser.error("--json is only supported with inspect and list")
    if args.operation in profile_operations and not args.profile:
        parser.error("--profile is required for deploy and lifecycle operations")
    if args.operation not in profile_operations and args.profile:
        parser.error("--profile is only supported with deploy and lifecycle operations")
    if args.monitoring_key and not args.monitoring_host:
        parser.error("--monitoring-key requires --monitoring-host")
    if args.monitoring_host and args.operation not in ("deploy", "remove"):
        parser.error("--monitoring-host is only supported with deploy and remove")
    try:
        configure_logging()
        if args.target:
            target = load_target(args.target)
        else:
            target = Target(host=args.host, provider=args.provider or DEFAULT_PROVIDER,
                            port=args.port or 22, key=args.key, known_hosts=args.known_hosts)
        logger.info("Starting %s operation for %s (provider %s)", args.operation, target.host,
                    target.provider)
        profile = load_profile(args.profile) if args.profile else None
        monitor = None
        if args.monitoring_host:
            if not profile.private_bind_ip:
                raise ValueError("--monitoring-host requires private_bind_ip in the deployment profile")
            monitor = SSHHost(args.monitoring_host, args.monitoring_key)
        host = create_provider(target)
        vast = target.provider == "vastai"
        if args.operation == "list":
            deployments = host.deployment_list()
            logger.info("Found %d inference-platform deployment(s)", len(deployments))
            if args.json:
                print(json.dumps({"host": target.host, "deployments": deployments}, indent=2))
            else:
                show_deployments(deployments)
            return 0
        if args.operation in lifecycle_operations:
            return run_lifecycle(host, args.operation, target.host, profile, monitor,
                                 kind="process" if vast else "container")
        report = host.inspect()
        proposal = host.plan(report)
        logger.debug("Host inspection report: %s", report)
        if args.json:
            print(json.dumps({"host": target.host, "report": report,
                              "blockers": proposal.blockers, "actions": proposal.actions}, indent=2))
        else:
            show_report(report, proposal)
        if args.operation == "inspect":
            return 2 if proposal.blockers else 0
        if args.operation == "deploy":
            if proposal.blockers or proposal.actions != ([] if vast else ["verify-gpu"]):
                print("Deployment requires a prepared host; run inspect/prepare first.", file=sys.stderr)
                return 2
            facts = host.deployment_inspect(profile.as_dict())
            blockers = host.deployment_blockers(facts, profile)
            logger.info("Deployment preflight found %d blocker(s)", len(blockers))
            logger.debug("Deployment preflight facts: %s", facts)
            show_deployment(profile, blockers)
            if blockers:
                return 2
            if monitor is not None:
                print(f"Will register the Prometheus target {profile.private_bind_ip}:{profile.port} "
                      f"on {monitor.target} after readiness checks.")
            if vast:
                print("\nThe existing container image digest cannot be independently verified here. "
                      "Confirm it matches the profile before proceeding.")
                print("Deployment downloads the public model, starts a vLLM process and runs "
                      "identity and generation checks. Restart is manual.")
            else:
                print("\nDeployment downloads the pinned image and public model, starts a persistent "
                      "container, and runs identity and generation checks.")
            print("Use only on a dedicated host; this workflow does not support shared GPU hosts.")
            if not sys.stdin.isatty():
                print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
                return 2
            confirmation = f"deploy {target.host} {profile.name}"
            if input(f"Type '{confirmation}' to proceed: ").strip() != confirmation:
                print("Cancelled; nothing changed.")
                return 0
            latest = host.inspect()
            updated = host.plan(latest)
            latest_facts = host.deployment_inspect(profile.as_dict())
            latest_blockers = host.deployment_blockers(latest_facts, profile)
            if (updated.blockers or updated.actions != proposal.actions
                    or latest["root"] != report["root"]
                    or latest["checks"]["gpu"] != report["checks"]["gpu"]
                    or latest_blockers):
                show_report(latest, updated)
                show_deployment(profile, latest_blockers)
                print("Host readiness changed; review and rerun deploy.", file=sys.stderr)
                return 2
            readiness = host.deploy(profile.as_dict())
            logger.info("Deployment readiness checks passed for %s", profile.name)
            print(f"\nDeployment ready: {readiness['model']} at {readiness['endpoint']}")
            print("Model identity check and generation smoke test passed.")
            if profile.private_bind_ip:
                print(f"Direct metrics: http://{profile.private_bind_ip}:{profile.port}/metrics")
            else:
                print("The endpoint is reachable only from the host; use an SSH tunnel for access.")
            if monitor is not None:
                return update_monitoring_target(monitor, "add", profile)
            return 0
        if proposal.blockers:
            return 2
        if vast:
            print("\nVast container prerequisites are available; no packages were installed and no model was deployed.")
            return 0
        print("\nPreparation can install packages, change Docker configuration and restart Docker.")
        print("Use only on a dedicated host. GPU verification may download a CUDA image.")
        if not sys.stdin.isatty():
            print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
            return 2
        if input(f"Type 'prepare {target.host}' to proceed: ").strip() != f"prepare {target.host}":
            print("Cancelled; nothing changed.")
            return 0
        latest = host.inspect()
        updated = host.plan(latest)
        if (updated.blockers or updated.actions != proposal.actions
                or latest["root"] != report["root"]
                or latest["checks"]["gpu"] != report["checks"]["gpu"]):
            show_report(latest, updated)
            print("Host readiness/plan changed; review and rerun prepare.", file=sys.stderr)
            return 2
        host.prepare(proposal.actions, report["root"])
        logger.info("Host preparation actions completed")
        final = host.inspect()
        final_plan = host.plan(final)
        show_report(final, final_plan)
        if final_plan.blockers or final_plan.actions != ["verify-gpu"]:
            print("Post-preparation inspection failed; host is not ready.", file=sys.stderr)
            return 2
        print("\nHost runtime ready: Docker GPU verification passed. No model was deployed.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        logger.error("%s operation failed: %s", args.operation, error)
        logger.debug("Operation failure details", exc_info=True)
        if isinstance(error, subprocess.CalledProcessError) and error.stdout:
            print(error.stdout.strip(), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        logger.warning("%s operation interrupted", args.operation)
        print("\nInterrupted; if preparation or deployment had started, inspect the host before retrying.",
              file=sys.stderr)
        return 130
