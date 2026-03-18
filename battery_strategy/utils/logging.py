from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import sys


_LOG_NAME = "battery_strategy"


@dataclass(slots=True)
class LoggingArtifacts:
    run_id: str
    log_file: Path
    warnings_errors_file: Path


def get_logger(name: str | None = None) -> logging.Logger:
    logger_name = _LOG_NAME if not name else f"{_LOG_NAME}.{name}"
    return logging.getLogger(logger_name)


def init_logging(output_dir: str | Path, *, run_id: str | None = None) -> tuple[logging.Logger, LoggingArtifacts]:
    output_path = Path(output_dir)
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{resolved_run_id}.log"
    warning_file = logs_dir / f"pipeline_{resolved_run_id}.warnings_errors.log"

    logger = logging.getLogger(_LOG_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
            logger.handlers = []
        else:
            return logger, LoggingArtifacts(
                run_id=resolved_run_id,
                log_file=log_file,
                warnings_errors_file=warning_file,
            )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    warnings_handler = logging.FileHandler(warning_file, encoding="utf-8")
    warnings_handler.setLevel(logging.WARNING)
    warnings_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.addHandler(warnings_handler)

    return logger, LoggingArtifacts(
        run_id=resolved_run_id,
        log_file=log_file,
        warnings_errors_file=warning_file,
    )
