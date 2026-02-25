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
        self._input_stream = sys.stdin
        self._color = _supports_color(self._stream)
        self._palette = _Palette()
        self._last_multiselect_unavailable_reason: str | None = None

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

    def last_multiselect_unavailable_reason(self) -> str | None:
        return self._last_multiselect_unavailable_reason

    def multi_select_with_start(
        self,
        *,
        title: str,
        options: list[tuple[str, str]],
        start_label: str = "开始",
        cancel_label: str = "取消",
    ) -> list[str] | None:
        self._last_multiselect_unavailable_reason = None

        if not options:
            return []

        stdout_is_tty = bool(hasattr(self._stream, "isatty") and self._stream.isatty())
        stdin_is_tty = bool(hasattr(self._input_stream, "isatty") and self._input_stream.isatty())
        if not (stdout_is_tty and stdin_is_tty):
            self._last_multiselect_unavailable_reason = "stdin/stdout 不是 TTY"
            return None

        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import Layout
            from prompt_toolkit.layout.containers import Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style
        except Exception as exc:  # noqa: BLE001
            self._last_multiselect_unavailable_reason = f"缺少依赖 prompt_toolkit（{exc.__class__.__name__}）"
            return None

        state = {"cursor": 0, "mode": "list"}
        selected: set[int] = set()

        def _render():
            total = len(options)
            cursor = state["cursor"]
            mode = state["mode"]

            terminal_lines = shutil.get_terminal_size((92, 30)).lines
            visible_count = max(6, terminal_lines - 12)
            start = max(0, min(cursor - visible_count + 1, max(0, total - visible_count)))
            end = min(total, start + visible_count)

            fragments: list[tuple[str, str]] = []
            fragments.append(("class:title", f"{title}\n"))
            fragments.append(("class:hint", "上下键移动 | Enter 选中/取消 | Tab 切换到按钮\n"))
            fragments.append(("class:hint", f"已选 {len(selected)} / {total}\n\n"))

            if start > 0:
                fragments.append(("class:hint", "...\n"))

            for idx in range(start, end):
                value, label = options[idx]
                _ = value
                marker = "x" if idx in selected else " "
                pointer = ">" if mode == "list" and idx == cursor else " "
                line_style = "class:cursor" if mode == "list" and idx == cursor else ""
                fragments.append((line_style, f"{pointer} [{marker}] {label}\n"))

            if end < total:
                fragments.append(("class:hint", "...\n"))

            fragments.append(("", "\n"))
            start_style = "class:button.focus" if mode == "start" else "class:button"
            cancel_style = "class:button.focus" if mode == "cancel" else "class:button"
            fragments.append((start_style, f"[ {start_label} ]"))
            fragments.append(("", "   "))
            fragments.append((cancel_style, f"[ {cancel_label} ]"))
            fragments.append(("", "\n"))
            return fragments

        kb = KeyBindings()

        @kb.add("up")
        def _up(event) -> None:
            if state["mode"] != "list":
                return
            if state["cursor"] > 0:
                state["cursor"] -= 1
                event.app.invalidate()

        @kb.add("down")
        def _down(event) -> None:
            if state["mode"] != "list":
                return
            if state["cursor"] < len(options) - 1:
                state["cursor"] += 1
                event.app.invalidate()

        @kb.add("tab")
        def _tab(event) -> None:
            mode_order = ["list", "start", "cancel"]
            current_idx = mode_order.index(state["mode"])
            state["mode"] = mode_order[(current_idx + 1) % len(mode_order)]
            event.app.invalidate()

        @kb.add("s-tab")
        def _shift_tab(event) -> None:
            mode_order = ["list", "start", "cancel"]
            current_idx = mode_order.index(state["mode"])
            state["mode"] = mode_order[(current_idx - 1) % len(mode_order)]
            event.app.invalidate()

        def _toggle_current() -> None:
            idx = state["cursor"]
            if idx in selected:
                selected.remove(idx)
            else:
                selected.add(idx)

        @kb.add(" ")
        def _space(event) -> None:
            if state["mode"] != "list":
                return
            _toggle_current()
            event.app.invalidate()

        @kb.add("enter")
        def _enter(event) -> None:
            mode = state["mode"]
            if mode == "list":
                _toggle_current()
                event.app.invalidate()
                return
            if mode == "start":
                result = [value for idx, (value, _label) in enumerate(options) if idx in selected]
                event.app.exit(result=result)
                return
            event.app.exit(result=None)

        @kb.add("escape")
        @kb.add("c-c")
        def _cancel(event) -> None:
            event.app.exit(result=None)

        style = Style.from_dict(
            {
                "title": "bold",
                "hint": "ansicyan",
                "cursor": "reverse",
                "button": "ansiblue",
                "button.focus": "reverse bold",
            }
        )

        app = Application(
            layout=Layout(Window(content=FormattedTextControl(_render), always_hide_cursor=True)),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=False,
        )

        try:
            return app.run()
        except Exception as exc:  # noqa: BLE001
            self._last_multiselect_unavailable_reason = (
                f"交互组件运行失败（{exc.__class__.__name__}: {exc}）"
            )
            return None
