from __future__ import annotations

import time

import numpy as np

from .vision import RecognitionResult


def screenshot() -> np.ndarray:
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("pyautogui is required for screen capture.") from exc

    image = pyautogui.screenshot()
    rgb = np.array(image)
    return rgb[:, :, ::-1].copy()


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
