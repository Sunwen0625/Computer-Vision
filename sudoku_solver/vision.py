from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .solver import Grid

Point = tuple[int, int]


class RecognitionError(RuntimeError):
    """Raised when the Sudoku screen cannot be recognized safely."""


@dataclass(frozen=True)
class RecognitionResult:
    grid: Grid
    cell_centers: dict[tuple[int, int], Point]
    digit_centers: dict[int, Point]
    board_bbox: tuple[int, int, int, int]
    confidence: dict[tuple[int, int], float]


@dataclass(frozen=True)
class _Template:
    digit: int
    image: np.ndarray
    center: Point


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RecognitionError(f"Cannot read image: {path}")
    return image


def recognize(image: np.ndarray) -> RecognitionResult:
    scale = _recognition_scale(image)
    if scale != 1:
        scaled = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        return _scale_result(_recognize_native(scaled), 1 / scale)

    return _recognize_native(image)


def _recognize_native(image: np.ndarray) -> RecognitionResult:
    board_bbox = _find_board_bbox(image)
    cell_centers = _cell_centers(board_bbox)
    visible_templates = _extract_digit_templates(image, board_bbox)
    templates = _complete_templates_from_references(visible_templates)
    digit_centers = {template.digit: template.center for template in visible_templates}
    grid, confidence = _recognize_grid(image, board_bbox, templates)

    return RecognitionResult(
        grid=grid,
        cell_centers=cell_centers,
        digit_centers=digit_centers,
        board_bbox=board_bbox,
        confidence=confidence,
    )


def _recognition_scale(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    min_side = min(height, width)
    if min_side >= 700:
        return 1
    return max(1, min(4, int(np.ceil(700 / max(min_side, 1)))))


def _scale_result(result: RecognitionResult, factor: float) -> RecognitionResult:
    x, y, w, h = result.board_bbox
    return RecognitionResult(
        grid=result.grid,
        cell_centers={
            cell: _scale_point(point, factor)
            for cell, point in result.cell_centers.items()
        },
        digit_centers={
            digit: _scale_point(point, factor)
            for digit, point in result.digit_centers.items()
        },
        board_bbox=(
            int(round(x * factor)),
            int(round(y * factor)),
            int(round(w * factor)),
            int(round(h * factor)),
        ),
        confidence=result.confidence,
    )


def offset_result(result: RecognitionResult, offset: Point) -> RecognitionResult:
    dx, dy = offset
    if dx == 0 and dy == 0:
        return result

    x, y, w, h = result.board_bbox
    return RecognitionResult(
        grid=result.grid,
        cell_centers={
            cell: (point[0] + dx, point[1] + dy)
            for cell, point in result.cell_centers.items()
        },
        digit_centers={
            digit: (point[0] + dx, point[1] + dy)
            for digit, point in result.digit_centers.items()
        },
        board_bbox=(x + dx, y + dy, w, h),
        confidence=result.confidence,
    )


def _scale_point(point: Point, factor: float) -> Point:
    return int(round(point[0] * factor)), int(round(point[1] * factor))


def _find_board_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([12, 45, 120]), np.array([45, 255, 255]))

    height, width = yellow.shape
    yellow[: int(height * 0.15), :] = 0
    yellow[int(height * 0.75) :, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=2)
    yellow = cv2.dilate(yellow, kernel, iterations=1)

    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.45 or h < width * 0.45:
            continue
        ratio = w / max(h, 1)
        if 0.85 <= ratio <= 1.15:
            candidates.append((w * h, x, y, w, h))

    if candidates:
        _, x, y, w, h = max(candidates)
        return _square_bbox(x, y, w, h, image.shape)

    rows = np.where(yellow.sum(axis=1) > width * 8)[0]
    cols = np.where(yellow.sum(axis=0) > height * 2)[0]
    if rows.size == 0 or cols.size == 0:
        raise RecognitionError("Could not find the yellow Sudoku board.")
    return _square_bbox(
        int(cols.min()),
        int(rows.min()),
        int(cols.max() - cols.min() + 1),
        int(rows.max() - rows.min() + 1),
        image.shape,
    )


def _square_bbox(
    x: int, y: int, width: int, height: int, image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    side = int(round((width + height) / 2))
    cx = x + width / 2
    cy = y + height / 2
    max_h, max_w = image_shape[:2]
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    side = min(side, max_w - left, max_h - top)
    return left, top, side, side


def _cell_centers(board_bbox: tuple[int, int, int, int]) -> dict[tuple[int, int], Point]:
    x, y, w, h = board_bbox
    cell_w = w / 9
    cell_h = h / 9
    return {
        (row, col): (
            int(round(x + (col + 0.5) * cell_w)),
            int(round(y + (row + 0.5) * cell_h)),
        )
        for row in range(9)
        for col in range(9)
    }


def _extract_digit_templates(
    image: np.ndarray, board_bbox: tuple[int, int, int, int]
) -> list[_Template]:
    height, width = image.shape[:2]
    _, board_y, _, board_h = board_bbox
    digit_components: list[tuple[int, int, int, int]] = []

    for start_y in _digit_search_starts(height, board_y, board_h):
        roi = image[start_y:, :]
        if roi.size == 0 or roi.shape[0] < 20:
            continue

        mask = _dark_mask(roi)
        components = _components(mask)
        candidate_components: list[tuple[int, int, int, int]] = []
        for x, y, w, h, area in components:
            absolute_y = y + start_y
            if absolute_y < board_y + board_h + board_h * 0.12:
                continue
            if h < height * 0.025 or w < width * 0.01:
                continue
            if area < 60:
                continue
            candidate_components.append((x, absolute_y, w, h))

        if len(candidate_components) >= 5:
            digit_components = candidate_components
            break

    if len(digit_components) < 1:
        raise RecognitionError(
            "Could not find the bottom 1-9 digit row. Capture the full game "
            "screen, including the number keyboard below the Sudoku board."
        )

    # The bottom keyboard digits are the nine widest-spread dark components
    # in the lower screen area. UI labels/icons sit above them and are ignored
    # by taking the median/lower text row and sorting left to right.
    digit_components.sort(key=lambda box: (box[1], box[0]))
    low_y = np.percentile([box[1] for box in digit_components], 70)
    digit_components = [box for box in digit_components if box[1] >= low_y - 15]
    digit_components.sort(key=lambda box: box[0])
    digit_components_by_slot = _components_by_digit_slot(digit_components, width)
    if not digit_components_by_slot:
        raise RecognitionError(
            "Could not isolate the bottom 1-9 digit row. Capture the full game "
            "screen with the number keyboard visible."
        )

    templates: list[_Template] = []
    for digit, (x, y, w, h) in sorted(digit_components_by_slot.items()):
        crop = _crop_with_padding(image, x, y, w, h, pad=8)
        template = _normalize_mask(_dark_mask(crop))
        templates.append(
            _Template(
                digit=digit,
                image=template,
                center=(int(round(x + w / 2)), int(round(y + h / 2))),
            )
        )
    return templates


def _complete_templates_from_references(templates: list[_Template]) -> list[_Template]:
    templates_by_digit = {template.digit: template for template in templates}
    missing = set(range(1, 10)) - set(templates_by_digit)
    if not missing:
        return templates

    for path in _reference_image_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        try:
            reference_templates = _extract_digit_templates(image, _find_board_bbox(image))
        except RecognitionError:
            continue

        for template in reference_templates:
            if template.digit in missing:
                templates_by_digit[template.digit] = _Template(
                    digit=template.digit,
                    image=template.image,
                    center=templates_by_digit.get(template.digit, template).center,
                )
        missing = set(range(1, 10)) - set(templates_by_digit)
        if not missing:
            break

    return [templates_by_digit[digit] for digit in sorted(templates_by_digit)]


def _reference_image_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd()
    candidates = [
        cwd / "gameOriginal.jpg",
        cwd / "game_tap.jpg",
        root / "gameOriginal.jpg",
        root / "game_tap.jpg",
    ]
    unique: list[Path] = []
    for path in candidates:
        if path not in unique and path.exists():
            unique.append(path)
    return unique


def _digit_search_starts(height: int, board_y: int, board_h: int) -> list[int]:
    starts = [
        board_y + board_h + 40,
        board_y + board_h + 100,
        int(height * 0.68),
        int(height * 0.74),
        int(height * 0.80),
    ]
    valid_starts = []
    for start in starts:
        clamped = max(0, min(start, height - 1))
        if clamped not in valid_starts:
            valid_starts.append(clamped)
    return valid_starts


def _components_by_digit_slot(
    components: list[tuple[int, int, int, int]], image_width: int
) -> dict[int, tuple[int, int, int, int]]:
    target_step = image_width / 9
    median_height = np.median([box[3] for box in components])
    selected: dict[int, tuple[float, tuple[int, int, int, int]]] = {}

    for box in components:
        x, y, w, h = box
        center_x = x + w / 2
        nearest_slot = round((center_x - target_step / 2) / target_step)
        digit = nearest_slot + 1
        if digit < 1 or digit > 9:
            continue

        slot_center = target_step * (nearest_slot + 0.5)
        score = abs(center_x - slot_center) + abs(h - median_height)
        if digit not in selected or score < selected[digit][0]:
            selected[digit] = (score, box)

    return {digit: box for digit, (_, box) in selected.items()}


def _recognize_grid(
    image: np.ndarray, board_bbox: tuple[int, int, int, int], templates: list[_Template]
) -> tuple[Grid, dict[tuple[int, int], float]]:
    x, y, w, h = board_bbox
    cell_w = w / 9
    cell_h = h / 9
    grid: Grid = []
    confidence: dict[tuple[int, int], float] = {}

    for row in range(9):
        values: list[int] = []
        for col in range(9):
            x1 = int(round(x + col * cell_w + cell_w * 0.16))
            y1 = int(round(y + row * cell_h + cell_h * 0.12))
            x2 = int(round(x + (col + 1) * cell_w - cell_w * 0.16))
            y2 = int(round(y + (row + 1) * cell_h - cell_h * 0.12))
            crop = image[y1:y2, x1:x2]
            mask = _digit_mask(crop)
            density = cv2.countNonZero(mask) / max(mask.size, 1)
            if density < 0.035:
                values.append(0)
                confidence[(row, col)] = 1.0
                continue

            digit_mask = _largest_digit_mask(mask)
            normalized = _normalize_mask(digit_mask)
            digit, score = _match_digit(normalized, templates)
            values.append(digit if score >= 0.38 else 0)
            confidence[(row, col)] = score
        grid.append(values)

    return grid, confidence


def _dark_mask(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        raise RecognitionError("Cannot process an empty image region.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sudoku digits are neutral dark gray. Yellow highlights and light grid
    # lines are excluded by combining value and saturation checks.
    return np.where((gray < 180) & (hsv[:, :, 1] < 130), 255, 0).astype(np.uint8)


def _digit_mask(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        raise RecognitionError("Cannot process an empty image region.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < 150) & (hsv[:, :, 1] < 120)
    blue = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] > 60) & (
        hsv[:, :, 2] > 120
    )
    return np.where(dark | blue, 255, 0).astype(np.uint8)


def _largest_digit_mask(mask: np.ndarray) -> np.ndarray:
    components = _components(mask)
    if not components:
        return mask
    x, y, w, h, _ = max(components, key=lambda item: item[4])
    return _crop_binary_with_padding(mask, x, y, w, h, pad=3)


def _match_digit(mask: np.ndarray, templates: list[_Template]) -> tuple[int, float]:
    best_digit = 0
    best_score = -1.0
    for template in templates:
        score = _mask_similarity(mask, template.image)
        if score > best_score:
            best_digit = template.digit
            best_score = score
    return best_digit, best_score


def _mask_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a > 0
    b_bool = b > 0
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _normalize_mask(mask: np.ndarray, size: int = 64) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    canvas = np.zeros((size, size), dtype=np.uint8)
    if xs.size == 0 or ys.size == 0:
        return canvas

    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = crop.shape
    scale = min((size - 10) / max(w, 1), (size - 10) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = np.where(resized > 80, 255, 0).astype(np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [
        (
            int(stats[index, cv2.CC_STAT_LEFT]),
            int(stats[index, cv2.CC_STAT_TOP]),
            int(stats[index, cv2.CC_STAT_WIDTH]),
            int(stats[index, cv2.CC_STAT_HEIGHT]),
            int(stats[index, cv2.CC_STAT_AREA]),
        )
        for index in range(1, count)
    ]


def _crop_with_padding(
    image: np.ndarray, x: int, y: int, w: int, h: int, pad: int
) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return image[y1:y2, x1:x2]


def _crop_binary_with_padding(
    mask: np.ndarray, x: int, y: int, w: int, h: int, pad: int
) -> np.ndarray:
    height, width = mask.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return mask[y1:y2, x1:x2]
