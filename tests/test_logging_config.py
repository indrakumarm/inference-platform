import logging
import os
import unittest
from unittest.mock import patch

from inference_platform.logging_config import (
    LOG_FORMAT,
    LOG_LEVEL_ENV,
    configure_logging,
    get_log_level,
)


class LoggingConfigurationTests(unittest.TestCase):
    def test_info_and_debug_levels_are_read_from_environment(self):
        for value in ("info", "DEBUG"):
            with self.subTest(value=value), patch.dict(os.environ, {LOG_LEVEL_ENV: value}):
                self.assertEqual(get_log_level(), value.upper())

    def test_unknown_level_is_an_explicit_configuration_error(self):
        with patch.dict(os.environ, {LOG_LEVEL_ENV: "verbose"}):
            with self.assertRaisesRegex(ValueError, LOG_LEVEL_ENV):
                get_log_level()

    def test_common_configuration_includes_module_name_and_stderr(self):
        root = logging.getLogger()
        with patch.dict(os.environ, {LOG_LEVEL_ENV: "DEBUG"}), \
                patch("inference_platform.logging_config.logging.basicConfig") as basic_config, \
                patch.object(root, "setLevel") as set_level:
            self.assertEqual(configure_logging(), "DEBUG")
        self.assertEqual(basic_config.call_args.kwargs["level"], "DEBUG")
        self.assertIn("%(name)s", basic_config.call_args.kwargs["format"])
        self.assertIs(basic_config.call_args.kwargs["stream"], __import__("sys").stderr)
        set_level.assert_called_once_with("DEBUG")


if __name__ == "__main__":
    unittest.main()
