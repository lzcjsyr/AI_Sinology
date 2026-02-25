from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable


def _supports_color(stream) -> bool:
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return bool(hasattr(stream, "isatty") and stream.isatty() and os.getenv("TERM") not in {None, "dumb"})


@dataclass(frozen=True)
class _Palette:
    reset: str = "\033[0m"
    bold: str = "\033[1m"
    dim: str = "\033[2m"
    cyan: str = "\033[36m"
    green: str = "\033[32m"
    yellow: str = "\033[33m"
    red: str = "\033[31m"


class CLIUI:
    def __init__(self) -> None:
        self._stream = sys.stdout
        self._color = _supports_color(self._stream)
        self._palette = _Palette()

    def _style(self, text: str, tone: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
        if not self._color:
            return text
        tokens: list[str] = []
        if bold:
            tokens.append(self._palette.bold)
        if dim:
            tokens.append(self._palette.dim)
        if tone:
            tokens.append(getattr(self._palette, tone))
        if not tokens:
            return text
        return f"{''.join(tokens)}{text}{self._palette.reset}"

    def _width(self) -> int:
        columns = shutil.get_terminal_size((92, 30)).columns
        return max(72, min(columns, 108))

    def _divider(self, char: str = "=", tone: str = "cyan") -> str:
        return self._style(char * self._width(), tone)

    def header(self, title: str, subtitle: str | None = None) -> None:
        print()
        print(self._divider("="))
        print(self._style(title, "cyan", bold=True))
        if subtitle:
            print(self._style(subtitle, dim=True))
        print(self._divider("="))
        print()

    def section(self, title: str) -> None:
        print()
        print(self._style(f"## {title}", "cyan", bold=True))
        print(self._style("-" * min(42, self._width()), tone="cyan", dim=True))
        print()

    def menu(self, title: str, options: Iterable[tuple[str, str]]) -> None:
        self.section(title)
        for key, label in options:
            print(f"{self._style(f'[{key}]', 'cyan', bold=True)}  {label}")
        print()

    def list_items(self, items: Iterable[str], *, prefix: str = "-") -> None:
        for item in items:
            print(f"{self._style(prefix, 'cyan')} {item}")
        print()

    def key_value(self, key: str, value: str) -> None:
        print(f"{self._style(key + ':', 'cyan', bold=True)} {value}")

    def info(self, message: str) -> None:
        print()
        print(f"{self._style('[INFO]', 'cyan', bold=True)} {message}")
        print()

    def success(self, message: str) -> None:
        print()
        print(f"{self._style('[OK]', 'green', bold=True)} {message}")
        print()

    def warning(self, message: str) -> None:
        print()
        print(f"{self._style('[WARN]', 'yellow', bold=True)} {message}")
        print()

    def error(self, message: str) -> None:
        print()
        print(f"{self._style('[ERROR]', 'red', bold=True)} {message}")
        print()

    def prompt(self, message: str) -> str:
        return input(f"{self._style('> ', 'cyan', bold=True)}{message} ").strip()

