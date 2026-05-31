from __future__ import annotations

Grid = list[list[int]]


class SudokuError(ValueError):
    """Raised when a puzzle is invalid or cannot be solved uniquely."""


def solve(grid: Grid) -> Grid:
    """Return the unique solution for a 9x9 Sudoku grid.

    Empty cells must be represented by 0. The function validates the input,
    rejects contradictory puzzles, and rejects puzzles with multiple solutions.
    """
    work = _copy_and_validate(grid)
    solutions: list[Grid] = []
    _search(work, solutions, limit=2)

    if not solutions:
        raise SudokuError("Puzzle has no solution.")
    if len(solutions) > 1:
        raise SudokuError("Puzzle has more than one solution.")
    return solutions[0]


def is_valid_solution(grid: Grid) -> bool:
    expected = set(range(1, 10))
    return (
        len(grid) == 9
        and all(len(row) == 9 for row in grid)
        and all(set(row) == expected for row in grid)
        and all({grid[row][col] for row in range(9)} == expected for col in range(9))
        and all(
            {
                grid[row][col]
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            }
            == expected
            for box_row in range(0, 9, 3)
            for box_col in range(0, 9, 3)
        )
    )


def _copy_and_validate(grid: Grid) -> Grid:
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise SudokuError("Puzzle must be a 9x9 grid.")

    work = [[int(value) for value in row] for row in grid]
    if any(value < 0 or value > 9 for row in work for value in row):
        raise SudokuError("Puzzle values must be between 0 and 9.")

    for row in range(9):
        for col in range(9):
            value = work[row][col]
            if value == 0:
                continue
            work[row][col] = 0
            if value not in _candidates(work, row, col):
                raise SudokuError(f"Contradiction at row {row + 1}, col {col + 1}.")
            work[row][col] = value

    return work


def _search(grid: Grid, solutions: list[Grid], limit: int) -> None:
    if len(solutions) >= limit:
        return

    cell = _best_empty_cell(grid)
    if cell is None:
        solutions.append([row[:] for row in grid])
        return

    row, col, candidates = cell
    for value in candidates:
        grid[row][col] = value
        _search(grid, solutions, limit)
        grid[row][col] = 0
        if len(solutions) >= limit:
            return


def _best_empty_cell(grid: Grid) -> tuple[int, int, list[int]] | None:
    best: tuple[int, int, list[int]] | None = None
    for row in range(9):
        for col in range(9):
            if grid[row][col] != 0:
                continue
            candidates = _candidates(grid, row, col)
            if not candidates:
                return row, col, []
            if best is None or len(candidates) < len(best[2]):
                best = row, col, candidates
    return best


def _candidates(grid: Grid, row: int, col: int) -> list[int]:
    used = set(grid[row])
    used.update(grid[r][col] for r in range(9))

    box_row = row - row % 3
    box_col = col - col % 3
    used.update(
        grid[r][c]
        for r in range(box_row, box_row + 3)
        for c in range(box_col, box_col + 3)
    )

    return [value for value in range(1, 10) if value not in used]
