from __future__ import annotations

import argparse
import json
from pathlib import Path

from .automation import fill_solution, screenshot
from .solver import SudokuError, is_valid_solution, solve
from .vision import RecognitionError, read_image, recognize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recognize, solve, and optionally auto-fill a Sudoku screen."
    )
    parser.add_argument("--auto", action="store_true", help="click cells and digits to fill")
    parser.add_argument("--dry-run", action="store_true", help="print recognition output only")
    parser.add_argument("--image", type=Path, help="recognize from an image instead of screen")
    parser.add_argument("--delay", type=float, default=0.08, help="seconds between clicks")
    args = parser.parse_args(argv)

    if args.image and args.auto and not args.dry_run:
        print("error: --auto can only click against a live screen capture, not --image.")
        return 1

    try:
        image = read_image(args.image) if args.image else screenshot()
        result = recognize(image)
        solution = solve(result.grid)
    except (RecognitionError, SudokuError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1

    _print_report(result.grid, solution, result.cell_centers, result.digit_centers)

    if not is_valid_solution(solution):
        print("error: solver returned an invalid solution")
        return 1

    if args.auto and not args.dry_run:
        fill_solution(result, solution, max(args.delay, 0.0))
    elif not args.dry_run:
        print("No clicks performed. Pass --auto to fill, or --dry-run to inspect only.")

    return 0


def _print_report(
    grid: list[list[int]],
    solution: list[list[int]],
    cell_centers: dict[tuple[int, int], tuple[int, int]],
    digit_centers: dict[int, tuple[int, int]],
) -> None:
    print("recognized grid:")
    print(_format_grid(grid))
    print("\nsolution:")
    print(_format_grid(solution))
    print("\ndigit centers:")
    print(json.dumps({str(k): v for k, v in digit_centers.items()}, ensure_ascii=False))
    empties = {
        f"{row + 1},{col + 1}": cell_centers[(row, col)]
        for row in range(9)
        for col in range(9)
        if grid[row][col] == 0
    }
    print("\nempty cell centers:")
    print(json.dumps(empties, ensure_ascii=False))


def _format_grid(grid: list[list[int]]) -> str:
    return "\n".join(
        " ".join(str(value) if value else "." for value in row) for row in grid
    )
