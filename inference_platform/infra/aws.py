from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import uuid

from ..providers import Target
from .config import AWSConfig
from .state import write_json


logger = logging.getLogger(__name__)
_MANAGED = "inference-platform:managed"
_LIVE_STATES = ["pending", "running", "stopping", "stopped", "shutting-down"]


@contextmanager
def aws_errors():
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as error:
        raise RuntimeError("AWS infra requires boto3: install with python3 -m pip install -e '.[aws]'") from error
    try:
        yield
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"AWS API failed: {error}") from error


def create_aws(config: AWSConfig):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as error:
        raise RuntimeError("AWS infra requires boto3: install with python3 -m pip install -e '.[aws]'") from error
    session = boto3.Session(profile_name=config.profile, region_name=config.region)
    options = Config(connect_timeout=10, read_timeout=30,
                     retries={"mode": "standard", "total_max_attempts": 3})
    account = session.client("sts", config=options).get_caller_identity()["Account"]
    return AWSInfra(config, session.client("ec2", config=options),
                    session.client("ssm", config=options), account)


class AWSInfra:
    """Named EC2 resources; all workload operations remain outside this adapter."""

    def __init__(self, config: AWSConfig, ec2, ssm, account: str):
        self.config = config
        self.ec2 = ec2
        self.ssm = ssm
        identity = f"{account}/{config.region}/{config.project}/{config.name}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.directory = Path(config.state_dir).expanduser() / "aws" / digest
        self.state_path = self.directory / "instance.json"
        self.target_path = self.directory / "target.json"

    def tags(self) -> list[dict[str, str]]:
        return [{"Key": "Name", "Value": self.config.name},
                {"Key": "Project", "Value": self.config.project},
                {"Key": _MANAGED, "Value": "true"}]

    def read_state(self) -> dict:
        if not self.state_path.exists():
            return {}
        data = json.loads(self.state_path.read_text())
        if (not isinstance(data, dict) or set(data) != {"request", "instance_id"} or
                (data["request"] is not None and
                 (not isinstance(data["request"], dict) or
                  not isinstance(data["request"].get("ClientToken"), str))) or
                (data["instance_id"] is None and data["request"] is None) or
                (data["instance_id"] is not None and not isinstance(data["instance_id"], str))):
            raise ValueError(f"Invalid infra state: {self.state_path}; do not retry creation with lost state.")
        return data

    def _check_owned(self, instance: dict) -> None:
        actual = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
        if any(actual.get(tag["Key"]) != tag["Value"] for tag in self.tags()):
            raise RuntimeError("Instance ownership tags changed; refusing to manage it.")

    def status(self) -> dict | None:
        filters = [{"Name": f"tag:{tag['Key']}", "Values": [tag["Value"]]}
                   for tag in self.tags()]
        filters.append({"Name": "instance-state-name", "Values": _LIVE_STATES})
        instances = []
        for page in self.ec2.get_paginator("describe_instances").paginate(Filters=filters):
            instances.extend(instance for reservation in page["Reservations"]
                             for instance in reservation["Instances"])
        if len(instances) > 1:
            raise RuntimeError("Multiple owned instances have this name; refusing ambiguous selection.")
        saved = self.read_state().get("instance_id")
        if instances:
            instance = instances[0]
            self._check_owned(instance)
            if saved and saved != instance["InstanceId"]:
                raise RuntimeError("Cloud instance differs from the recorded ID; refusing to switch resources.")
            pending = self.read_state()
            if (pending and pending["instance_id"] is None and
                    instance.get("ClientToken") != pending["request"]["ClientToken"]):
                raise RuntimeError("Discovered instance is not the persisted launch; refusing to switch resources.")
            return instance
        if saved:
            response = self.ec2.describe_instances(InstanceIds=[saved])
            instance = response["Reservations"][0]["Instances"][0]
            self._check_owned(instance)
            return instance
        return None

    def launch_request(self) -> dict:
        saved = self.read_state()
        if saved and saved["instance_id"] is None:
            tags = next((spec["Tags"] for spec in saved["request"].get("TagSpecifications", [])
                         if spec["ResourceType"] == "instance"), [])
            if tags != self.tags():
                raise RuntimeError("Persisted launch ownership differs from the config; refusing to create.")
            logger.info("Resuming the persisted EC2 launch request")
            return saved["request"]
        image_id = self.config.ami_id
        if self.config.ami_ssm:
            image_id = self.ssm.get_parameter(Name=self.config.ami_ssm)["Parameter"]["Value"]
        images = self.ec2.describe_images(ImageIds=[image_id])["Images"]
        if len(images) != 1 or images[0].get("RootDeviceType") != "ebs":
            raise ValueError("Infra requires an EBS-backed AMI")
        image = images[0]
        if image.get("Architecture") != "x86_64":
            raise ValueError("The SSH workload provider currently requires an x86_64 AMI")
        root = image["RootDeviceName"]
        minimum = next((mapping.get("Ebs", {}).get("VolumeSize", 0)
                        for mapping in image.get("BlockDeviceMappings", [])
                        if mapping["DeviceName"] == root), 0)
        if self.config.root_gb < minimum:
            raise ValueError(f"root_gb must be at least the AMI root volume size ({minimum} GiB)")
        request = {
            "ClientToken": str(uuid.uuid4()), "ImageId": image_id,
            "InstanceType": self.config.instance_type, "MinCount": 1, "MaxCount": 1,
            "KeyName": self.config.key_name, "SubnetId": self.config.subnet_id,
            "SecurityGroupIds": self.config.security_group_ids,
            "MetadataOptions": {"HttpTokens": "required"},
            "BlockDeviceMappings": [{"DeviceName": root, "Ebs": {
                "VolumeSize": self.config.root_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}],
            "TagSpecifications": [{"ResourceType": kind, "Tags": self.tags()}
                                  for kind in ("instance", "volume")],
        }
        if self.config.instance_profile:
            request["IamInstanceProfile"] = {"Name": self.config.instance_profile}
        if self.config.market == "spot":
            request["InstanceMarketOptions"] = {
                "MarketType": "spot", "SpotOptions": {
                    "SpotInstanceType": "persistent", "InstanceInterruptionBehavior": "stop"}}
        return request

    def _wait(self, state: str, instance_id: str) -> None:
        self.ec2.get_waiter(f"instance_{state}").wait(
            InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 120})

    def _refresh(self, instance_id: str) -> dict:
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        instance = response["Reservations"][0]["Instances"][0]
        self._check_owned(instance)
        return instance

    def _remember(self, instance: dict) -> None:
        saved = self.read_state()
        if not saved:
            write_json(self.state_path, {"request": None, "instance_id": instance["InstanceId"]})
            return
        if saved and saved["instance_id"] is None:
            if instance.get("ClientToken") != saved["request"]["ClientToken"]:
                raise RuntimeError("Discovered instance is not the persisted launch; refusing to switch resources.")
            saved["instance_id"] = instance["InstanceId"]
            write_json(self.state_path, saved)

    def apply(self, operation: str, expected: dict | None, request: dict | None = None) -> dict:
        current = self.status()
        signature = lambda item: (item["InstanceId"], item["State"]["Name"]) if item else None
        if signature(current) != signature(expected):
            raise RuntimeError("Instance state changed after confirmation; review and rerun.")
        if operation == "up" and (current is None or current["State"]["Name"] == "terminated"):
            if request is None:
                raise ValueError("A reviewed launch request is required")
            if current and current.get("SpotInstanceRequestId"):
                raise RuntimeError("Run down to cancel the previous persistent spot request before creating again.")
            self.target_path.unlink(missing_ok=True)
            write_json(self.state_path, {"request": request, "instance_id": None})
            result = self.ec2.run_instances(**request)
            instance_id = result["Instances"][0]["InstanceId"]
            write_json(self.state_path, {"request": request, "instance_id": instance_id})
            logger.info("Created %s; state saved at %s", instance_id, self.state_path)
            self._wait("running", instance_id)
            return self._refresh(instance_id)
        if current is None:
            raise RuntimeError("No owned instance exists for this name; run infra aws up first.")
        self._remember(current)
        instance_id, state = current["InstanceId"], current["State"]["Name"]
        if operation in ("up", "start"):
            if state == "stopping":
                self._wait("stopped", instance_id)
                current = self._refresh(instance_id)
                state = current["State"]["Name"]
            if state == "stopped":
                self.ec2.start_instances(InstanceIds=[instance_id])
            elif state not in ("pending", "running"):
                raise RuntimeError(f"Cannot start an instance in state {state}")
            self._wait("running", instance_id)
            return self._refresh(instance_id)
        if operation == "stop":
            if state == "running":
                self.target_path.unlink(missing_ok=True)
                self.ec2.stop_instances(InstanceIds=[instance_id])
            elif state not in ("stopping", "stopped"):
                raise RuntimeError(f"Cannot stop an instance in state {state}")
            self._wait("stopped", instance_id)
            self.target_path.unlink(missing_ok=True)
            return self._refresh(instance_id)
        if operation == "down":
            self.target_path.unlink(missing_ok=True)
            if current.get("SpotInstanceRequestId"):
                self.ec2.cancel_spot_instance_requests(
                    SpotInstanceRequestIds=[current["SpotInstanceRequestId"]])
            if state not in ("shutting-down", "terminated"):
                self.ec2.terminate_instances(InstanceIds=[instance_id])
            self._wait("terminated", instance_id)
            self.state_path.unlink(missing_ok=True)
            return self._refresh(instance_id)
        raise ValueError(f"Unsupported infra operation: {operation}")

    def resolve(self, instance: dict) -> Target:
        self._check_owned(instance)
        if instance["State"]["Name"] != "running":
            raise RuntimeError("Target resolution requires a running instance")
        address = (instance.get("PrivateIpAddress") if self.config.private_ip else
                   instance.get("PublicIpAddress") or instance.get("PrivateIpAddress"))
        if not address:
            raise RuntimeError("Running instance has no usable IP address")
        return Target(host=f"{self.config.ssh_user}@{address}", provider="ssh",
                      key=self.config.key, known_hosts=self.config.known_hosts)

    def validate_export(self, export: Path | None) -> None:
        protected = [self.state_path, self.directory / "lock"]
        protected.extend(Path(path).expanduser() for path in (
            self.config.key, self.config.known_hosts) if path)
        if export is not None and export.resolve() in [path.resolve() for path in protected]:
            raise ValueError("Target export cannot overwrite infra state, its lock or SSH credentials")

    def write_target(self, instance: dict, export: Path | None = None) -> Path:
        self.validate_export(self.target_path)
        self.validate_export(export)
        data = asdict(self.resolve(instance))
        self._remember(instance)
        write_json(self.target_path, data)
        if export is not None and export.resolve() != self.target_path.resolve():
            write_json(export, data)
            return export
        return self.target_path
