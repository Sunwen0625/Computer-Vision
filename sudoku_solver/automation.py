from __future__ import annotations

import ctypes
import time

import numpy as np

from .vision import Point, RecognitionResult


def screenshot(window_title: str | None = None) -> tuple[np.ndarray, Point]:
    _enable_dpi_awareness()

    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("pyautogui is required for screen capture.") from exc

    if window_title:
        window = _find_window(window_title)
        if window.isMinimized:
            window.restore()
            time.sleep(0.2)
        try:
            window.activate()
            time.sleep(0.2)
        except Exception:
            pass

        left = int(window.left)
        top = int(window.top)
        width = int(window.width)
        height = int(window.height)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Window has invalid size: {window_title}")
        image = _grab_window(left, top, width, height)
        offset = (left, top)
    else:
        image = pyautogui.screenshot()
        offset = (0, 0)

    rgb = np.array(image)
    return rgb[:, :, ::-1].copy(), offset


def _grab_window(left: int, top: int, width: int, height: int):
    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError("Pillow is required for window capture.") from exc

    if _is_windows():
        min_x, min_y = _virtual_screen_origin()
        desktop = ImageGrab.grab(all_screens=True)
        box = (
            left - min_x,
            top - min_y,
            left - min_x + width,
            top - min_y + height,
        )
        return desktop.crop(box)

    return ImageGrab.grab(bbox=(left, top, left + width, top + height))


def list_window_titles() -> list[str]:
    _enable_dpi_awareness()
    try:
        import pygetwindow as gw
    except ImportError as exc:
        raise RuntimeError("pygetwindow is required for listing windows.") from exc

    return sorted({title for title in gw.getAllTitles() if title.strip()})


def _is_windows() -> bool:
    return hasattr(ctypes, "windll")


def _virtual_screen_origin() -> Point:
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(76)), int(user32.GetSystemMetrics(77))


def fill_solution(result: RecognitionResult, solution: list[list[int]], delay: float) -> None:
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("pyautogui is required for auto clicking.") from exc

    pyautogui.FAILSAFE = True
    for row in range(9):
        for col in range(9):
            if result.grid[row][col] != 0:
                continue

            digit = solution[row][col]
            if digit not in result.digit_centers:
                raise RuntimeError(
                    f"Digit {digit} is required at row {row + 1}, col {col + 1}, "
                    "but its bottom keyboard button is not visible."
                )

            cell_x, cell_y = result.cell_centers[(row, col)]
            digit_x, digit_y = result.digit_centers[digit]
            pyautogui.click(cell_x, cell_y)
            time.sleep(delay)
            pyautogui.click(digit_x, digit_y)
            time.sleep(delay)


def _find_window(title_part: str):
    _enable_dpi_awareness()
    try:
        import pygetwindow as gw
    except ImportError as exc:
        raise RuntimeError("pygetwindow is required for window capture.") from exc

    matches = [
        window
        for window in gw.getAllWindows()
        if title_part.lower() in window.title.lower()
    ]
    if not matches:
        available = ", ".join(list_window_titles()[:10])
        raise RuntimeError(
            f"Could not find a window matching {title_part!r}. "
            f"Use --list-windows to see titles. First titles: {available}"
        )

    return max(matches, key=lambda window: window.width * window.height)


def _enable_dpi_awareness() -> None:
    if not _is_windows():
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
