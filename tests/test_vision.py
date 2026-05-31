from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from sudoku_solver.solver import is_valid_solution, solve
from sudoku_solver.vision import read_image, recognize


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GRID = [
    [6, 5, 9, 4, 3, 1, 0, 2, 0],
    [0, 3, 2, 8, 9, 5, 0, 1, 6],
    [8, 1, 4, 2, 6, 7, 0, 0, 0],
    [5, 0, 0, 3, 8, 2, 9, 7, 4],
    [2, 4, 0, 5, 7, 9, 6, 3, 1],
    [9, 7, 3, 6, 1, 0, 2, 0, 0],
    [3, 0, 6, 1, 5, 8, 0, 0, 0],
    [4, 8, 0, 0, 0, 3, 1, 0, 0],
    [1, 2, 0, 0, 0, 0, 0, 5, 8],
]


@pytest.mark.parametrize("filename", ["gameOriginal.jpg", "game_tap.jpg"])
def test_recognizes_reference_images(filename):
    result = recognize(read_image(ROOT / filename))

    assert result.grid == EXPECTED_GRID
    assert len(result.cell_centers) == 81
    assert set(result.digit_centers) == set(range(1, 10))
    assert is_valid_solution(solve(result.grid))
