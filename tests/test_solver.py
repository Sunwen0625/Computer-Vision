import pytest

from sudoku_solver.solver import SudokuError, is_valid_solution, solve


PUZZLE = [
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


def test_solver_finds_valid_unique_solution():
    solution = solve(PUZZLE)

    assert is_valid_solution(solution)
    assert solution[0] == [6, 5, 9, 4, 3, 1, 8, 2, 7]


def test_solver_rejects_contradiction():
    puzzle = [row[:] for row in PUZZLE]
    puzzle[0][1] = 6

    with pytest.raises(SudokuError):
        solve(puzzle)
