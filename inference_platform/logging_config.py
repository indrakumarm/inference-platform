import logging
import os
import sys


LOG_LEVEL_ENV = "INFERENCE_PLATFORM_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def get_log_level():
    level = os.environ.get(LOG_LEVEL_ENV, "WARNING").strip().upper()
    if level not in _LOG_LEVELS:
        choices = ", ".join(sorted(_LOG_LEVELS))
        raise ValueError(f"{LOG_LEVEL_ENV} must be one of: {choices}")
    return level


def configure_logging():
    level = get_log_level()
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)
    logging.getLogger().setLevel(level)
    return level
