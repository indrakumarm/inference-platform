import argparse
import json
import logging
import math
from pathlib import Path
import sys

from ..logging_config import configure_logging
from .aws import aws_errors, create_aws
from .config import load_config
from .state import resource_lock
from .vast import create_vast, load_vast_config


logger = logging.getLogger(__name__)


def infra_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="inference_platform infra", description="Manage infrastructure, not vLLM workloads.")
    providers = parser.add_subparsers(dest="provider", required=True)
    aws = providers.add_parser("aws", help="Named AWS EC2 GPU instances via the Python SDK")
    aws.add_argument("operation", choices=("up", "start", "stop", "down", "status", "resolve"))
    aws.add_argument("--config", required=True, type=Path, help="AWS infrastructure JSON config")
    aws.add_argument("--target", type=Path, help="Export target JSON (up/start/resolve only)")
    aws.add_argument("--json", action="store_true", help="Machine-readable status or resolve output")
    vast = providers.add_parser("vastai", help="Vast.ai container instances")
    vast.add_argument("operation", choices=("search", "up", "start", "stop", "down", "status", "resolve"))
    vast.add_argument("--config", required=True, type=Path, help="Vast infrastructure JSON config")
    vast.add_argument("--offer", type=int, help="Offer ID selected from search (up only)")
    vast.add_argument("--target", type=Path, help="Export target JSON (resolve only)")
    vast.add_argument("--json", action="store_true", help="Machine-readable search or status output")
    args = parser.parse_args(argv)
    if args.provider == "vastai":
        if ((args.operation == "up" and (args.offer is None or args.offer <= 0))
                or (args.operation != "up" and args.offer is not None)):
            parser.error("up requires a positive --offer; --offer is not supported by other operations")
        if args.target and args.operation != "resolve":
            parser.error("--target is only supported by resolve")
        if args.json and args.operation not in ("search", "status"):
            parser.error("--json is only supported by search and status")
        if args.target and args.target.resolve() == args.config.resolve():
            parser.error("--target cannot overwrite --config")
        try:
            configure_logging()
            return run_vast(args, load_vast_config(args.config))
        except (EOFError, KeyboardInterrupt):
            print("Interrupted; check Vast status before retrying. Resources may still be billable.",
                  file=sys.stderr)
            return 130
        except (OSError, ValueError, RuntimeError) as error:
            logger.error("Vast %s failed: %s", args.operation, error)
            print("Check Vast status before retrying; no rollback is attempted.", file=sys.stderr)
            return 1
    if args.target and args.operation not in ("up", "start", "resolve"):
        parser.error("--target is only supported by up, start and resolve")
    if args.json and args.operation not in ("status", "resolve"):
        parser.error("--json is only supported by status and resolve")
    if args.target and args.target.resolve() == args.config.resolve():
        parser.error("--target cannot overwrite --config")
    try:
        configure_logging()
        config = load_config(args.config)
        with aws_errors():
            return run_aws(args, config)
    except (EOFError, KeyboardInterrupt):
        logger.warning("Infra operation interrupted")
        print("Interrupted; inspect infra status before retrying. Resources may still be billable.",
              file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("Infra %s failed: %s", args.operation, error)
        print("Check infra status before retrying; no rollback is attempted.", file=sys.stderr)
        return 1


def run_vast(args, config):
    provider = create_vast(config)
    if args.operation == "search":
        offers = provider.search()
        rows = [{key: offer.get(key) for key in ("id", "gpu_name", "num_gpus", "dph_total")}
                for offer in offers]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['id']}: {row['num_gpus']}x {row['gpu_name']}, ${row['dph_total']}/hour")
        return 0
    with resource_lock(provider.directory):
        state = provider.read_state()
        instance = provider.status(state)
        if args.operation == "status":
            summary = None if instance is None else {
                key: instance.get(key) for key in ("id", "actual_status", "intended_status", "label")}
            if args.json:
                print(json.dumps({"name": config.name, "instance": summary}, indent=2))
            else:
                print(f"{config.name}: " + (f"{summary['id']} [{summary['actual_status']}]"
                                             if summary else "no recorded instance"))
            return 0
        if args.operation == "resolve":
            if instance is None:
                raise RuntimeError("No recorded Vast instance exists")
            if args.target and args.target.exists():
                raise RuntimeError("Target file already exists; choose a new path")
            path = provider.write_target(instance, args.target)
            print(f"Target written: {path}")
            print("Verify the SSH host key through a trusted channel before connecting.")
            return 0
        if args.operation == "up":
            if instance is not None:
                raise RuntimeError("Instance already exists; use start or status")
            matching = [o for o in provider.search() if o.get("id") == args.offer]
            if len(matching) != 1:
                raise RuntimeError("Selected offer is unavailable or not in the current search results")
            offer = matching[0]
            price = offer.get("dph_total")
            if (isinstance(price, bool) or not isinstance(price, (int, float))
                    or not math.isfinite(price) or price < 0):
                raise RuntimeError("Selected offer has no valid hourly price")
            print(f"Create Vast container: {offer.get('num_gpus')}x {offer.get('gpu_name')}, "
                  f"${price}/hour, disk {config.disk_gb} GB, image {config.image}.")
            print("Compute and storage charges apply; SSH proxy is enabled. No vLLM port is exposed.")
            print("The workload adapter verifies prerequisites after creation; image digest "
                  "cannot be independently verified inside the container.")
        elif instance is None:
            raise RuntimeError("No recorded Vast instance exists")
        else:
            print(f"{args.operation}: instance {instance['id']} [{instance.get('actual_status')}]")
            print("This affects the whole instance. Check Vast's current compute and storage charges.")
            if args.operation == "down":
                print("Destroying an instance is irreversible and deletes its instance data.")
        if not sys.stdin.isatty():
            print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
            return 2
        confirmation = f"{args.operation} {config.name}"
        if input(f"Type '{confirmation}' to proceed: ").strip() != confirmation:
            print("Cancelled; nothing changed.")
            return 0
        if args.operation == "up":
            matching = [o for o in provider.search() if o.get("id") == args.offer]
            if len(matching) != 1 or matching[0].get("dph_total") != price:
                raise RuntimeError("Offer availability or price changed; review and retry")
            instance_id = provider.create(args.offer)
            print(f"Vast creation requested: {instance_id}. Check status before resolving SSH.")
        else:
            provider.change(args.operation, state, instance)
            print("Vast accepted the request; check status for the actual state.")
        return 0


def run_aws(args, config):
    provider = create_aws(config)
    provider.validate_export(args.target)
    with resource_lock(provider.directory):
        instance = provider.status()
        if args.operation == "status":
            pending = instance is None and bool(provider.read_state())
            result = {"name": config.name, "region": config.region, "instance": instance,
                      "launch_pending": pending}
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            elif instance:
                print(f"{config.name}: {instance['InstanceId']} [{instance['State']['Name']}]")
            elif pending:
                print(f"{config.name}: launch outcome unknown; rerun up with its saved request.")
            else:
                print(f"{config.name}: no owned instance found")
            return 0
        if args.operation == "resolve":
            if instance is None:
                raise RuntimeError("No owned instance exists for this name")
            path = provider.write_target(instance, args.target)
            if args.json:
                print(path.read_text(), end="")
            else:
                print(f"Target written: {path}")
            return 0
        request = None
        if args.operation == "up" and (instance is None or instance["State"]["Name"] == "terminated"):
            request = provider.launch_request()
            root_gb = request["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"]
            market = "spot" if "InstanceMarketOptions" in request else "ondemand"
            print(f"Create {config.name}: {request['InstanceType']} in {config.region}, "
                  f"AMI {request['ImageId']}, root {root_gb} GiB, market {market}.")
            print(f"Subnet: {request['SubnetId']}; security groups: "
                  f"{', '.join(request['SecurityGroupIds'])}; key pair: {request['KeyName']}.")
        elif instance:
            print(f"{args.operation}: {config.name} ({instance['InstanceId']}, "
                  f"{instance['State']['Name']})")
        else:
            raise RuntimeError("No owned instance exists for this name; run infra aws up first.")
        if args.operation == "down":
            print("Terminates the instance and deletes its root volume/model cache. "
                  "Other attached volumes follow their AWS DeleteOnTermination settings.")
        elif args.operation == "stop":
            print("Stops the machine and every workload on it. Retained storage remains billable.")
        else:
            print("Compute and storage charges apply. No host preparation or model deployment is performed.")
        if not sys.stdin.isatty():
            print("Interactive confirmation is required; nothing changed.", file=sys.stderr)
            return 2
        confirmation = f"{args.operation} {config.name}"
        if input(f"Type '{confirmation}' to proceed: ").strip() != confirmation:
            print("Cancelled; nothing changed.")
            return 0
        result = provider.apply(args.operation, instance, request)
        if args.operation in ("up", "start"):
            path = provider.write_target(result, args.target)
            print(f"Instance running: {result['InstanceId']}. Target written: {path}")
            print("Verify the SSH host key before connecting; running is not proof of SSH/GPU readiness.")
            print(f"Next: python3 -m inference_platform inspect --target {path}")
        else:
            print(f"{result['InstanceId']}: {result['State']['Name']}")
        return 0
