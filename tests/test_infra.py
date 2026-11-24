import contextlib
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from inference_platform import cli
from inference_platform.infra.aws import AWSInfra, aws_errors
from inference_platform.infra.config import AWSConfig, load_config
from inference_platform.infra.state import resource_lock, write_json
from inference_platform.providers import load_target
from inference_platform.ssh import SSHHost


def config_data():
    return {"name": "lab-a", "region": "us-east-1", "instance_type": "g6.xlarge",
            "subnet_id": "subnet-abc", "security_group_ids": ["sg-abc"],
            "key_name": "lab", "root_gb": 100, "ami_id": "ami-abc"}


def instance(provider, state="running", **fields):
    return {"InstanceId": "i-abc", "State": {"Name": state},
            "Tags": provider.tags(), "PublicIpAddress": "192.0.2.10",
            "PrivateIpAddress": "10.0.1.10", **fields}


class InfraConfigTests(unittest.TestCase):
    def test_missing_optional_sdk_reports_actionable_error(self):
        with patch.dict("sys.modules", {"botocore": None, "botocore.exceptions": None}):
            with self.assertRaisesRegex(RuntimeError, r"pip install -e '\.\[aws\]'"):
                with aws_errors():
                    self.fail("Missing SDK was ignored")

    def test_config_resolves_paths_and_rejects_bad_or_ambiguous_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aws.json"
            path.write_text(json.dumps({**config_data(), "key": "keys/lab.pem"}))
            config = load_config(path)
            self.assertEqual(config.key, str((path.parent / "keys/lab.pem").resolve()))
            for changes in ({"ami_ssm": "/aws/image"}, {"root_gb": True},
                            {"security_group_ids": ["sg-abc", "sg-abc"]},
                            {"market": "fallback"}, {"private_ip": "yes"},
                            {"name": "../../other"}, {"unknown": 1}):
                path.write_text(json.dumps({**config_data(), **changes}))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load_config(path)

    def test_resource_lock_prevents_concurrent_local_operations_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with resource_lock(root):
                with self.assertRaisesRegex(RuntimeError, "Another infra operation"):
                    with resource_lock(root):
                        self.fail("Lock admitted another operation")
            with resource_lock(root):
                pass


class InfraLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = AWSConfig(**config_data(), state_dir=self.temporary.name)
        self.ec2 = Mock()
        self.provider = AWSInfra(self.config, self.ec2, Mock(), "123456789012")

    def pages(self, *instances):
        self.ec2.get_paginator.return_value.paginate.return_value = [
            {"Reservations": [{"Instances": list(instances)}]}]

    def test_discovery_is_owned_paginated_and_ambiguity_blocks_mutation(self):
        one = instance(self.provider)
        self.ec2.get_paginator.return_value.paginate.return_value = [
            {"Reservations": []}, {"Reservations": [{"Instances": [one]}]}]
        self.assertEqual(self.provider.status(), one)
        filters = self.ec2.get_paginator.return_value.paginate.call_args.kwargs["Filters"]
        self.assertIn({"Name": "tag:inference-platform:managed", "Values": ["true"]}, filters)
        self.pages(one, {**one, "InstanceId": "i-def"})
        with self.assertRaisesRegex(RuntimeError, "Multiple owned"):
            self.provider.apply("stop", one)
        self.ec2.stop_instances.assert_not_called()

    def test_state_change_and_recorded_id_mismatch_prevent_mutation(self):
        old = instance(self.provider)
        self.pages(instance(self.provider, "stopped"))
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            self.provider.apply("stop", old)
        self.ec2.stop_instances.assert_not_called()
        write_json(self.provider.state_path,
                   {"request": {"ClientToken": "saved"}, "instance_id": "i-def"})
        with self.assertRaisesRegex(RuntimeError, "recorded ID"):
            self.provider.status()

    def test_launch_resolves_ssm_and_enforces_image_requirements_without_market_fallback(self):
        self.provider.config = replace(self.config, ami_id=None, ami_ssm="/aws/image",
                                       market="spot", instance_profile="gpu-role")
        self.provider.ssm.get_parameter.return_value = {"Parameter": {"Value": "ami-def"}}
        image = {"RootDeviceType": "ebs", "RootDeviceName": "/dev/sda1", "Architecture": "x86_64",
                 "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 20}}]}
        self.ec2.describe_images.return_value = {"Images": [image]}
        request = self.provider.launch_request()
        self.assertEqual(request["ImageId"], "ami-def")
        self.assertEqual(request["IamInstanceProfile"], {"Name": "gpu-role"})
        self.assertEqual(request["InstanceMarketOptions"]["SpotOptions"],
                         {"SpotInstanceType": "persistent", "InstanceInterruptionBehavior": "stop"})
        for change in ({"Architecture": "arm64"}, {"RootDeviceType": "instance-store"},
                       {"BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 200}}]}):
            self.ec2.describe_images.return_value = {"Images": [{**image, **change}]}
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.provider.launch_request()
        self.ec2.run_instances.assert_not_called()

    def test_discovered_launch_must_match_saved_client_token(self):
        write_json(self.provider.state_path,
                   {"request": {"ClientToken": "saved"}, "instance_id": None})
        self.pages(instance(self.provider, ClientToken="different"))
        with self.assertRaisesRegex(RuntimeError, "not the persisted launch"):
            self.provider.status()

    def test_stop_waits_and_invalidates_only_managed_target(self):
        running = instance(self.provider)
        self.pages(running)
        write_json(self.provider.target_path, {"host": "ubuntu@192.0.2.10"})
        export = Path(self.temporary.name) / "export.json"
        export.write_text("snapshot")
        self.ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [instance(self.provider, "stopped")]}]}
        result = self.provider.apply("stop", running)
        self.ec2.stop_instances.assert_called_once_with(InstanceIds=["i-abc"])
        self.assertEqual(result["State"]["Name"], "stopped")
        self.assertFalse(self.provider.target_path.exists())
        self.assertEqual(export.read_text(), "snapshot")

    def test_start_refreshes_address_and_checks_ownership(self):
        stopped = instance(self.provider, "stopped")
        self.pages(stopped)
        running = instance(self.provider, PublicIpAddress="192.0.2.20")
        self.ec2.describe_instances.return_value = {"Reservations": [{"Instances": [running]}]}
        result = self.provider.apply("start", stopped)
        self.ec2.start_instances.assert_called_once_with(InstanceIds=["i-abc"])
        target = load_target(self.provider.write_target(result))
        self.assertEqual(target.host, "ubuntu@192.0.2.20")
        self.assertEqual(target.provider, "ssh")
        self.assertIn("StrictHostKeyChecking=yes", SSHHost(target.host).command)
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            self.provider.resolve({**running, "Tags": []})

    def test_down_cancels_persistent_spot_before_termination_and_retains_state_on_failure(self):
        running = instance(self.provider, SpotInstanceRequestId="sir-abc")
        self.pages(running)
        write_json(self.provider.state_path,
                   {"request": {"ClientToken": "saved"}, "instance_id": "i-abc"})
        self.ec2.get_waiter.return_value.wait.side_effect = RuntimeError("wait timed out")
        with self.assertRaisesRegex(RuntimeError, "wait timed out"):
            self.provider.apply("down", running)
        operations = [call[0] for call in self.ec2.mock_calls]
        self.assertLess(operations.index("cancel_spot_instance_requests"),
                        operations.index("terminate_instances"))
        self.assertTrue(self.provider.state_path.exists())
        self.ec2.get_waiter.return_value.wait.side_effect = None
        self.ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [instance(self.provider, "terminated")]}]}
        self.provider.apply("down", running)
        self.assertFalse(self.provider.state_path.exists())

    def test_resolve_private_network_and_does_not_overwrite_host_keys_or_state(self):
        keys = Path(self.temporary.name) / "known_hosts"
        keys.write_text("trusted host key\n")
        self.provider.config = replace(self.config, private_ip=True, known_hosts=str(keys))
        target = load_target(self.provider.write_target(instance(self.provider)))
        self.assertEqual(target.host, "ubuntu@10.0.1.10")
        self.assertEqual(keys.read_text(), "trusted host key\n")
        transport = SSHHost(target.host, known_hosts=target.known_hosts)
        self.assertIn(f"UserKnownHostsFile={keys}", transport.command)
        with self.assertRaisesRegex(ValueError, "overwrite infra state"):
            self.provider.write_target(instance(self.provider), self.provider.state_path)
        with self.assertRaisesRegex(ValueError, "SSH credentials"):
            self.provider.write_target(instance(self.provider), keys)


class InfraCLITests(unittest.TestCase):
    def invoke(self, operation, provider, directory, answer="up lab-a", terminal=True, extra=()):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(config_data()))
        output = io.StringIO()
        with patch("inference_platform.infra.cli.create_aws", return_value=provider), \
                patch("inference_platform.infra.cli.aws_errors", return_value=contextlib.nullcontext()), \
                patch("sys.stdin.isatty", return_value=terminal), \
                patch("builtins.input", return_value=answer) as prompt, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = cli.main(["infra", "aws", operation, "--config", str(path), *extra])
        return code, prompt, output.getvalue()

    def test_confirmation_blocks_noninteractive_cancel_and_routes_without_workload_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(directory=Path(directory))
            provider.status.return_value = None
            provider.launch_request.return_value = {
                "InstanceType": "g6.xlarge", "ImageId": "ami-abc",
                "SubnetId": "subnet-abc", "SecurityGroupIds": ["sg-abc"], "KeyName": "lab",
                "BlockDeviceMappings": [{"Ebs": {"VolumeSize": 100}}]}
            for terminal, answer, expected in ((False, "up lab-a", 2), (True, "no", 0)):
                with self.subTest(terminal=terminal):
                    code, _, output = self.invoke("up", provider, directory, answer, terminal)
                    self.assertEqual(code, expected)
                    provider.apply.assert_not_called()
                    provider.write_target.assert_not_called()
                    self.assertNotIn("Instance running", output)

    def test_up_emits_target_only_after_success_and_never_reports_failed_launch_as_running(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(directory=Path(directory))
            provider.status.return_value = None
            request = {"InstanceType": "g6.xlarge", "ImageId": "ami-abc",
                       "SubnetId": "subnet-abc", "SecurityGroupIds": ["sg-abc"], "KeyName": "lab",
                       "BlockDeviceMappings": [{"Ebs": {"VolumeSize": 100}}]}
            provider.launch_request.return_value = request
            provider.apply.return_value = {"InstanceId": "i-abc"}
            provider.write_target.return_value = Path(directory) / "target.json"
            code, prompt, output = self.invoke("up", provider, directory)
            self.assertEqual(code, 0)
            provider.apply.assert_called_once_with("up", None, request)
            prompt.assert_called_once()
            self.assertIn("running is not proof of SSH/GPU readiness", output)
            provider.reset_mock()
            provider.apply.side_effect = RuntimeError("capacity unavailable")
            code, _, output = self.invoke("up", provider, directory)
            self.assertEqual(code, 1)
            provider.write_target.assert_not_called()
            self.assertNotIn("Instance running", output)

    def test_status_and_resolve_do_not_prompt_or_mutate_aws(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Mock(directory=Path(directory))
            provider.status.return_value = {"InstanceId": "i-abc", "State": {"Name": "running"}}
            target_path = Path(directory) / "target.json"
            write_json(target_path, {"provider": "ssh", "host": "ubuntu@192.0.2.10"})
            provider.write_target.return_value = target_path
            for operation in ("status", "resolve"):
                code, prompt, output = self.invoke(operation, provider, directory, extra=("--json",))
                self.assertEqual(code, 0)
                json.loads(output)
                prompt.assert_not_called()
            provider.apply.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("boto3"), "Install .[aws] for SDK contract tests")
class AWSSDKTests(unittest.TestCase):
    def test_cli_creation_exports_target_consumed_by_existing_workload_cli(self):
        import boto3
        from botocore.stub import Stubber

        ec2 = boto3.client("ec2", region_name="us-east-1",
                           aws_access_key_id="test", aws_secret_access_key="test")
        with tempfile.TemporaryDirectory() as directory, Stubber(ec2) as stubber:
            known_hosts = Path(directory) / "known_hosts"
            known_hosts.write_text("trusted key\n")
            data = {**config_data(), "state_dir": directory, "known_hosts": str(known_hosts)}
            config_path = Path(directory) / "aws.json"
            config_path.write_text(json.dumps(data))
            provider = AWSInfra(load_config(config_path), ec2, Mock(), "123456789012")
            stubber.add_response("describe_images", {"Images": [{
                "RootDeviceType": "ebs", "RootDeviceName": "/dev/sda1", "Architecture": "x86_64"}]},
                {"ImageIds": ["ami-abc"]})
            request = provider.launch_request()
            write_json(provider.state_path, {"request": request, "instance_id": None})
            stubber.add_response("describe_instances", {"Reservations": []})
            stubber.add_response("describe_instances", {"Reservations": []})
            stubber.add_response("run_instances", {"Instances": [{"InstanceId": "i-abc"}]}, request)
            running = instance(provider, ClientToken=request["ClientToken"])
            for _ in range(2):
                stubber.add_response("describe_instances", {"Reservations": [{"Instances": [running]}]},
                                     {"InstanceIds": ["i-abc"]})
            target = Path(directory) / "target.json"
            with patch("inference_platform.infra.cli.create_aws", return_value=provider), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("builtins.input", return_value="up lab-a"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["infra", "aws", "up", "--config", str(config_path),
                                           "--target", str(target)]), 0)
            with patch("inference_platform.ssh.subprocess.run", return_value=Mock(stdout="[]")) as ssh, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["list", "--target", str(target), "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue()), {"host": "ubuntu@192.0.2.10", "deployments": []})
            self.assertIn("StrictHostKeyChecking=yes", ssh.call_args.args[0])
            self.assertIn(f"UserKnownHostsFile={known_hosts}", ssh.call_args.args[0])
            self.assertEqual(known_hosts.read_text(), "trusted key\n")
            stubber.assert_no_pending_responses()

    def test_ambiguous_create_failure_replays_exact_saved_request_with_same_token(self):
        import boto3
        from botocore.exceptions import ClientError
        from botocore.stub import Stubber

        ec2 = boto3.client("ec2", region_name="us-east-1",
                           aws_access_key_id="test", aws_secret_access_key="test")
        with tempfile.TemporaryDirectory() as directory, Stubber(ec2) as stubber:
            provider = AWSInfra(AWSConfig(**config_data(), state_dir=directory),
                                ec2, Mock(), "123456789012")
            stubber.add_response("describe_images", {"Images": [{
                "ImageId": "ami-abc", "RootDeviceType": "ebs", "RootDeviceName": "/dev/sda1",
                "Architecture": "x86_64",
                "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 20}}]}]},
                {"ImageIds": ["ami-abc"]})
            request = provider.launch_request()
            self.assertEqual(request["MetadataOptions"], {"HttpTokens": "required"})
            self.assertTrue(request["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"])
            self.assertEqual(request["MaxCount"], 1)
            filters = [{"Name": f"tag:{tag['Key']}", "Values": [tag["Value"]]}
                       for tag in provider.tags()]
            filters.append({"Name": "instance-state-name",
                            "Values": ["pending", "running", "stopping", "stopped", "shutting-down"]})
            for retry in (False, True):
                stubber.add_response("describe_instances", {"Reservations": []}, {"Filters": filters})
                if not retry:
                    stubber.add_client_error("run_instances", service_error_code="RequestTimeout",
                                             expected_params=request)
                    with self.assertRaises(ClientError):
                        provider.apply("up", None, request)
                    self.assertEqual(provider.launch_request(), request)
                    provider.config = replace(provider.config, root_gb=200)
                    self.assertEqual(provider.launch_request(), request)
                else:
                    running = instance(provider, ClientToken=request["ClientToken"])
                    stubber.add_response("run_instances", {"Instances": [{"InstanceId": "i-abc"}]}, request)
                    stubber.add_response("describe_instances", {"Reservations": [{"Instances": [running]}]},
                                         {"InstanceIds": ["i-abc"]})
                    stubber.add_response("describe_instances", {"Reservations": [{"Instances": [running]}]},
                                         {"InstanceIds": ["i-abc"]})
                    self.assertEqual(provider.apply("up", None, request)["InstanceId"], "i-abc")
            self.assertEqual(provider.read_state()["instance_id"], "i-abc")
            stubber.assert_no_pending_responses()


if __name__ == "__main__":
    unittest.main()
