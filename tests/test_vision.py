from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from sudoku_solver.solver import is_valid_solution, solve
from sudoku_solver.vision import RecognitionError, offset_result, read_image, recognize


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


def test_missing_bottom_digit_row_reports_recognition_error():
    image = read_image(ROOT / "gameOriginal.jpg")
    cropped = image[:1650, :]

    with pytest.raises(RecognitionError, match="bottom 1-9 digit row"):
        recognize(cropped)


def test_offset_result_translates_click_coordinates():
    result = recognize(read_image(ROOT / "gameOriginal.jpg"))
    shifted = offset_result(result, (100, 200))

    assert shifted.cell_centers[(0, 0)] == (
        result.cell_centers[(0, 0)][0] + 100,
        result.cell_centers[(0, 0)][1] + 200,
    )
    assert shifted.digit_centers[1] == (
        result.digit_centers[1][0] + 100,
        result.digit_centers[1][1] + 200,
    )
