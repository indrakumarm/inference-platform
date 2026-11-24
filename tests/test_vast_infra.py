import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from inference_platform.infra.cli import infra_main
from inference_platform.infra.vast import VastConfig, VastInfra, create_vast, load_vast_config


def profile_data():
    return {
        "name": "model", "image": "vllm/vllm-openai@sha256:" + "a" * 64,
        "model": "org/model", "model_revision": "b" * 40,
        "tokenizer": "org/model", "tokenizer_revision": "b" * 40,
        "gpu_indices": [0], "min_gpu_memory_gib": 16, "min_free_disk_gib": 40,
        "max_model_len": 4096, "dtype": "auto", "gpu_memory_utilization": 0.9,
        "port": 8000, "shm_size_gib": 8,
    }


class VastInfraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.config = VastConfig("lab", "vllm/vllm-openai@sha256:" + "a" * 64,
                                 100, "RTX 4090", 1, None, None, str(root), root / ".env")
        self.api = Mock()
        self.provider = VastInfra(self.config, self.api)

    def test_create_persists_unknown_outcome_before_call_and_never_retries(self):
        self.api.request.side_effect = RuntimeError("timeout")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.provider.create(123)
        state = self.provider.read_state()
        self.assertEqual(state["phase"], "creating")
        self.assertIsNone(state["instance_id"])
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            self.provider.status(state)
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            self.provider.create(123)
        self.assertEqual(self.api.request.call_count, 1)

    def test_create_uses_existing_image_and_no_exposed_model_port(self):
        self.api.request.return_value = {"success": True, "new_contract": 42}
        self.assertEqual(self.provider.create(123), 42)
        method, path, body = self.api.request.call_args.args
        self.assertEqual((method, path), ("PUT", "asks/123/"))
        self.assertEqual(body["image"], self.config.image)
        self.assertEqual(body["runtype"], "ssh_proxy")
        self.assertNotIn("env", body)
        self.assertEqual(self.provider.read_state()["instance_id"], 42)

    def test_api_key_environment_precedes_dotenv(self):
        self.config.env_file.write_text("VAST_API_KEY=from-file\n")
        with patch.dict("os.environ", {"VAST_API_KEY": "from-environment"}):
            self.assertEqual(create_vast(self.config).api.token, "from-environment")
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(create_vast(self.config).api.token, "from-file")

    def test_wrong_ownership_and_changed_state_prevent_mutation(self):
        state = {"name": "lab", "label": "managed", "offer_id": 123,
                 "instance_id": 42, "phase": "created"}
        self.api.request.return_value = {"instances": {"id": 42, "label": "other"}}
        with self.assertRaisesRegex(RuntimeError, "ownership"):
            self.provider.status(state)
        self.api.request.return_value = {
            "instances": {"id": 42, "label": "managed", "actual_status": "stopped"}}
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.provider.change("down", state, {"actual_status": "running"})
        self.assertEqual(self.api.request.call_args.args[0], "GET")

    def test_target_requires_running_and_valid_ssh_details(self):
        instance = {"actual_status": "loading", "ssh_host": "ssh1.vast.ai", "ssh_port": 1234}
        with self.assertRaisesRegex(RuntimeError, "not running"):
            self.provider.write_target(instance, None)
        instance["actual_status"] = "running"
        path = self.provider.write_target(instance, None)
        self.assertEqual(json.loads(path.read_text()),
                         {"provider": "vastai", "host": "root@ssh1.vast.ai", "port": 1234})

    def test_config_uses_existing_deployment_image(self):
        root = Path(self.temporary.name)
        profile = root / "deployment.json"
        profile.write_text(json.dumps(profile_data()))
        path = root / "vast.json"
        path.write_text(json.dumps({"name": "lab", "deployment": str(profile),
                                    "disk_gb": 100, "gpu_name": "RTX 4090", "num_gpus": 1}))
        config = load_vast_config(path)
        self.assertTrue(config.image.startswith("vllm/vllm-openai@sha256:"))
        path.write_text(json.dumps({"name": "lab", "deployment": str(profile),
                                    "disk_gb": 100, "gpu_name": "RTX 4090", "num_gpus": 2}))
        with self.assertRaisesRegex(ValueError, "GPU count"):
            load_vast_config(path)

    def test_noninteractive_creation_never_mutates(self):
        offer = {"id": 123, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.4}
        self.api.request.return_value = {"offers": [offer]}
        root = Path(self.temporary.name)
        profile = root / "deployment.json"
        profile.write_text(json.dumps(profile_data()))
        path = root / "vast.json"
        path.write_text(json.dumps({"name": "lab", "deployment": str(profile),
            "disk_gb": 100, "gpu_name": "RTX 4090", "num_gpus": 1, "state_dir": str(root)}))
        with patch("inference_platform.infra.cli.create_vast", return_value=self.provider), \
                patch("sys.stdin.isatty", return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(infra_main(["vastai", "up", "--config", str(path), "--offer", "123"]), 2)
        self.assertFalse(self.provider.state_path.exists())
        self.assertEqual(self.api.request.call_count, 1)

    def test_price_change_after_confirmation_prevents_creation(self):
        offer = {"id": 123, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.4}
        self.api.request.side_effect = [{"offers": [offer]},
                                        {"offers": [{**offer, "dph_total": 0.5}]}]
        root = Path(self.temporary.name)
        profile = root / "deployment.json"
        profile.write_text(json.dumps(profile_data()))
        path = root / "vast.json"
        path.write_text(json.dumps({"name": "lab", "deployment": str(profile),
                                    "disk_gb": 100, "gpu_name": "RTX 4090", "num_gpus": 1}))
        with patch("inference_platform.infra.cli.create_vast", return_value=self.provider), \
                patch("sys.stdin.isatty", return_value=True), \
                patch("builtins.input", return_value="up lab"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(infra_main(["vastai", "up", "--config", str(path), "--offer", "123"]), 1)
        self.assertFalse(self.provider.state_path.exists())
        self.assertEqual(self.api.request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
