from __future__ import annotations

import time

import numpy as np

from .vision import Point, RecognitionResult


def screenshot(window_title: str | None = None) -> tuple[np.ndarray, Point]:
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
        image = pyautogui.screenshot(region=(left, top, width, height))
        offset = (left, top)
    else:
        image = pyautogui.screenshot()
        offset = (0, 0)

    rgb = np.array(image)
    return rgb[:, :, ::-1].copy(), offset


def list_window_titles() -> list[str]:
    try:
        import pygetwindow as gw
    except ImportError as exc:
        raise RuntimeError("pygetwindow is required for listing windows.") from exc

    return sorted({title for title in gw.getAllTitles() if title.strip()})


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
            cell_x, cell_y = result.cell_centers[(row, col)]
            digit_x, digit_y = result.digit_centers[digit]
            pyautogui.click(cell_x, cell_y)
            time.sleep(delay)
            pyautogui.click(digit_x, digit_y)
            time.sleep(delay)


def _find_window(title_part: str):
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
