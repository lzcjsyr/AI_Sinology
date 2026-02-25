from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _supports_color(stream) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return bool(hasattr(stream, "isatty") and stream.isatty() and os.getenv("TERM") not in {None, "dumb"})


class _ConsoleFormatter(logging.Formatter):
    _RESET = "\033[0m"
    _COLORS = {
        logging.DEBUG: "\033[2m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    def __init__(self, *, use_color: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self._use_color = use_color

    def _colorize(self, text: str, levelno: int) -> str:
        if not self._use_color:
            return text
        color = self._COLORS.get(levelno, "")
        if not color:
            return text
        return f"{color}{text}{self._RESET}"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname.ljust(8)
        prefix = f"{timestamp} | {level} |"
        message = record.getMessage()

        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        elif record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        return f"{self._colorize(prefix, record.levelno)} {message}"


def setup_logger(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("thesis_agent")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_ConsoleFormatter(use_color=_supports_color(sys.stdout)))
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    return logger
