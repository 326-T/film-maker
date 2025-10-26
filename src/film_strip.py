"""Utilities for generating a film-strip style scrolling video from still images.

The script reads every supported image from an input directory, stitches the
frames into a single horizontal strip, and creates a MoviePy ``VideoClip`` that
scrolls through the strip sideways – mimicking a film reel moving past the
camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from moviepy import ImageClip, VideoClip

# Common image extensions – extend as needed.
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


@dataclass(frozen=True)
class FilmStripSettings:
    """Centralized configuration for film-strip generation."""

    top_input_dir: Path = Path("input/travel")
    bottom_input_dir: Path | None = None
    output_path: Path = Path("output/travel.mp4")
    target_height: int = 360
    viewport_width: int = 1920
    scroll_speed: float = 260.0
    fps: int = 30
    lead_in: float = 0.75
    lead_out: float = 0.75
    frame_margin: int = 0
    row_spacing: int = 0
    stripe_height: int = 70
    stripe_light_color: Tuple[int, int, int] = (235, 235, 235)
    stripe_dark_color: Tuple[int, int, int] = (26, 26, 26)
    perforation_size: int = 40
    background_color: Tuple[int, int, int] = (12, 12, 16)


SETTINGS = FilmStripSettings()


@dataclass(frozen=True)
class StripAssets:
    """Raw numpy arrays used to render the film strip."""

    strip_frame: np.ndarray
    perforation_band: np.ndarray
    strip_width: int
    row_height: int


@dataclass(frozen=True)
class LayoutMetrics:
    """Precomputed layout positions for composing the frame."""

    viewport_width: int
    frame_height: int
    stripe_height: int
    top_row_top: int
    top_image_top: int
    bottom_row_top: int
    bottom_image_top: int
    top_row_height: int
    bottom_row_height: int


@dataclass(frozen=True)
class ScrollPlan:
    """Stores scroll timing information for the dual film strips."""

    distance: int
    speed: float
    lead_in: float
    lead_out: float
    scroll_duration: float
    total_duration: float


def create_perforation_band(
    width: int,
    height: int,
    *,
    band_color: Tuple[int, int, int],
    perforation_color: Tuple[int, int, int],
    square_size: int,
) -> np.ndarray:
    """Return a band with repeating perforation squares along its width."""

    if square_size <= 0:
        msg = "square_size must be positive"
        raise ValueError(msg)

    band = np.empty((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        band[:, :, channel] = band_color[channel]

    usable_size = min(square_size, height)
    vertical_offset = (height - usable_size) // 2

    step = max(square_size * 2, 1)
    for x in range(0, width, step):
        x_end = min(x + square_size, width)
        band[vertical_offset : vertical_offset + usable_size, x:x_end, :] = (
            perforation_color
        )

    return band


def natural_key(path: Path) -> Tuple[Tuple[object, ...], str]:
    """Provide a natural sorting key based on the filename.

    Examples: ``10.jpg`` comes after ``2.jpg``.
    """

    import re

    def convert(segment: str) -> object:
        return int(segment) if segment.isdigit() else segment.lower()

    stem = path.stem
    return tuple(convert(s) for s in re.split(r"(\d+)", stem)), stem.lower()


def collect_image_paths(input_dir: Path) -> List[Path]:
    """Return sorted image paths within ``input_dir`` using natural order."""

    return sort_paths_naturally(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def sort_paths_naturally(paths: Sequence[Path] | Iterable[Path]) -> List[Path]:
    """Sort paths by human-friendly filename order with deterministic fallback."""

    def key(path: Path) -> Tuple[Tuple[object, ...], str, str, str]:
        natural, stem_lower = natural_key(path)
        return natural, stem_lower, path.suffix.lower(), str(path)

    return sorted(list(paths), key=key)


def split_paths_for_rows(paths: Sequence[Path]) -> Tuple[List[Path], List[Path]]:
    """Partition ``paths`` (already sorted) into odd/even slots for rows."""

    top_paths: List[Path] = []
    bottom_paths: List[Path] = []

    for index, path in enumerate(paths):
        if index % 2 == 0:  # 1-based odd indices
            top_paths.append(path)
        else:
            bottom_paths.append(path)

    return top_paths, bottom_paths


def log_row_assignments(
    top_paths: Sequence[Path], bottom_paths: Sequence[Path]
) -> None:
    """Print the sorted filenames assigned to the top/bottom rows."""

    def format_listing(label: str, items: Sequence[Path]) -> None:
        print(label)
        for path in items:
            print(f"  - {path.name}")

    format_listing("Top row (odd indices):", top_paths)
    format_listing("Bottom row (even indices, reversed for playback):", bottom_paths)


def load_resized_frames(
    image_paths: Sequence[Path], target_height: int
) -> Tuple[List[np.ndarray], int]:
    """Load images, resize them to a uniform target height, and return frames."""

    frames: List[np.ndarray] = []
    total_width = 0

    for path in image_paths:
        clip = ImageClip(str(path)).resized(height=target_height)
        frame = clip.get_frame(0)
        clip.close()

        # ``ImageClip`` ensures uint8 output, but guard just in case.
        if frame.dtype != np.uint8:
            frame = frame.astype("uint8")

        frames.append(frame)
        total_width += frame.shape[1]

    return frames, total_width


def build_strip(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate individual frames horizontally into one large image."""

    if not frames:
        msg = "No frames provided to build the strip."
        raise ValueError(msg)

    # All frames are resized to the same height in ``load_resized_frames``.
    return np.concatenate(frames, axis=1)


def load_strip_assets(
    image_paths: Sequence[Path],
    *,
    target_height: int,
    stripe_height: int,
    stripe_dark_color: Tuple[int, int, int],
    stripe_light_color: Tuple[int, int, int],
    perforation_size: int,
) -> StripAssets:
    """Load source images and supporting bands for rendering."""

    frames, strip_width = load_resized_frames(image_paths, target_height)
    strip_frame = build_strip(frames)
    perforation_band = create_perforation_band(
        strip_width,
        stripe_height,
        band_color=stripe_dark_color,
        perforation_color=stripe_light_color,
        square_size=perforation_size,
    )

    return StripAssets(
        strip_frame=strip_frame,
        perforation_band=perforation_band,
        strip_width=strip_width,
        row_height=strip_frame.shape[0],
    )


def calculate_layout(
    viewport_width: int,
    *,
    top_row_height: int,
    bottom_row_height: int,
    stripe_height: int,
    frame_margin: int,
    row_spacing: int,
) -> LayoutMetrics:
    """Compute vertical positions for the strip and perforation bands."""

    top_block_height = top_row_height + (stripe_height * 2)
    bottom_block_height = bottom_row_height + (stripe_height * 2)
    frame_height = (
        frame_margin * 2 + top_block_height + bottom_block_height + row_spacing
    )

    top_row_top = frame_margin
    top_image_top = top_row_top + stripe_height
    bottom_row_top = frame_margin + top_block_height + row_spacing
    bottom_image_top = bottom_row_top + stripe_height

    return LayoutMetrics(
        viewport_width=viewport_width,
        frame_height=frame_height,
        stripe_height=stripe_height,
        top_row_top=top_row_top,
        top_image_top=top_image_top,
        bottom_row_top=bottom_row_top,
        bottom_image_top=bottom_image_top,
        top_row_height=top_row_height,
        bottom_row_height=bottom_row_height,
    )


def calculate_scroll_plan(
    strip_width: int,
    viewport_width: int,
    *,
    scroll_speed: float,
    lead_in: float,
    lead_out: float,
) -> ScrollPlan:
    """Precompute scroll metrics for consistent frame generation."""

    distance = max(strip_width - viewport_width, 0)
    scroll_duration = distance / scroll_speed if distance else 0.0
    total_duration = lead_in + scroll_duration + lead_out

    return ScrollPlan(
        distance=distance,
        speed=scroll_speed,
        lead_in=lead_in,
        lead_out=lead_out,
        scroll_duration=scroll_duration,
        total_duration=total_duration,
    )


def scroll_offset(plan: ScrollPlan, t: float, *, reverse: bool) -> int:
    """Return the horizontal slice offset for a given timestamp."""

    if plan.distance == 0:
        return 0

    if t <= plan.lead_in:
        progress = 0.0
    elif t >= plan.lead_in + plan.scroll_duration:
        progress = float(plan.distance)
    else:
        progress = (t - plan.lead_in) * plan.speed

    progress = max(0.0, min(progress, float(plan.distance)))
    offset = float(plan.distance) - progress if reverse else progress
    offset_int = int(round(offset))
    return max(0, min(offset_int, plan.distance))


def build_frame_function(
    top_assets: StripAssets,
    bottom_assets: StripAssets,
    layout: LayoutMetrics,
    *,
    top_plan: ScrollPlan,
    bottom_plan: ScrollPlan,
    background_color: Tuple[int, int, int],
) -> Callable[[float], np.ndarray]:
    """Prepare the ``make_frame`` callable for MoviePy."""

    base_frame = np.full(
        (layout.frame_height, layout.viewport_width, 3),
        fill_value=background_color,
        dtype=np.uint8,
    )

    def make_frame(t: float) -> np.ndarray:
        frame = base_frame.copy()

        top_offset = scroll_offset(top_plan, t, reverse=False)
        bottom_offset = scroll_offset(bottom_plan, t, reverse=True)

        top_strip = top_assets.strip_frame[
            :, top_offset : top_offset + layout.viewport_width, :
        ].copy()
        bottom_strip = bottom_assets.strip_frame[
            :, bottom_offset : bottom_offset + layout.viewport_width, :
        ].copy()

        top_band = top_assets.perforation_band[
            :, top_offset : top_offset + layout.viewport_width, :
        ].copy()
        bottom_band = bottom_assets.perforation_band[
            :, bottom_offset : bottom_offset + layout.viewport_width, :
        ].copy()

        frame[layout.top_row_top : layout.top_row_top + layout.stripe_height, :, :] = (
            top_band
        )
        frame[
            layout.top_image_top : layout.top_image_top + layout.top_row_height, :, :
        ] = top_strip
        frame[
            layout.top_image_top + layout.top_row_height : layout.top_image_top
            + layout.top_row_height
            + layout.stripe_height,
            :,
            :,
        ] = top_band

        frame[
            layout.bottom_row_top : layout.bottom_row_top + layout.stripe_height, :, :
        ] = bottom_band
        frame[
            layout.bottom_image_top : layout.bottom_image_top
            + layout.bottom_row_height,
            :,
            :,
        ] = bottom_strip
        frame[
            layout.bottom_image_top + layout.bottom_row_height : layout.bottom_image_top
            + layout.bottom_row_height
            + layout.stripe_height,
            :,
            :,
        ] = bottom_band

        return frame

    return make_frame


def create_film_strip_video(
    top_input_dir: Path,
    output_path: Path,
    *,
    bottom_input_dir: Path | None = None,
    target_height: int,
    viewport_width: int | None,
    scroll_speed: float,
    fps: int,
    lead_in: float,
    lead_out: float,
    frame_margin: int,
    row_spacing: int,
    stripe_height: int,
    stripe_light_color: Tuple[int, int, int],
    stripe_dark_color: Tuple[int, int, int],
    perforation_size: int,
    background_color: Tuple[int, int, int],
) -> Path:
    """Generate a dual-row film-strip video with stylised borders.

    The top and bottom image folders are stitched into separate strips. The
    upper row scrolls left-to-right, while the lower row scrolls right-to-left.
    Each row is framed by a perforated stripe pattern to evoke the look of
    analogue film.
    """
    # This function expects positive values for the geometry/timing parameters;
    # we skip explicit validation and rely on the fixed configuration above.

    all_paths = collect_image_paths(top_input_dir)
    if bottom_input_dir and bottom_input_dir != top_input_dir:
        all_paths.extend(collect_image_paths(bottom_input_dir))

    if not all_paths:
        msg = "No supported images found in the provided input directories"
        raise FileNotFoundError(msg)

    all_paths = sort_paths_naturally(all_paths)
    top_paths, bottom_paths = split_paths_for_rows(all_paths)

    if not top_paths:
        msg = "At least one image is required for the top row"
        raise FileNotFoundError(msg)
    if not bottom_paths:
        msg = "At least two images are required to populate the bottom row"
        raise FileNotFoundError(msg)

    # The bottom strip scrolls from right-to-left, so reverse the frame order
    # to keep the chronological sequence consistent with the top row.
    bottom_paths = list(reversed(bottom_paths))

    log_row_assignments(top_paths, bottom_paths)

    top_assets = load_strip_assets(
        top_paths,
        target_height=target_height,
        stripe_height=stripe_height,
        stripe_dark_color=stripe_dark_color,
        stripe_light_color=stripe_light_color,
        perforation_size=perforation_size,
    )

    bottom_assets = load_strip_assets(
        bottom_paths,
        target_height=target_height,
        stripe_height=stripe_height,
        stripe_dark_color=stripe_dark_color,
        stripe_light_color=stripe_light_color,
        perforation_size=perforation_size,
    )

    max_view = viewport_width or min(top_assets.strip_width, bottom_assets.strip_width)
    effective_viewport = min(
        max_view, top_assets.strip_width, bottom_assets.strip_width
    )

    layout = calculate_layout(
        effective_viewport,
        top_row_height=top_assets.row_height,
        bottom_row_height=bottom_assets.row_height,
        stripe_height=stripe_height,
        frame_margin=frame_margin,
        row_spacing=row_spacing,
    )

    top_plan = calculate_scroll_plan(
        top_assets.strip_width,
        effective_viewport,
        scroll_speed=scroll_speed,
        lead_in=lead_in,
        lead_out=lead_out,
    )

    bottom_plan = calculate_scroll_plan(
        bottom_assets.strip_width,
        effective_viewport,
        scroll_speed=scroll_speed,
        lead_in=lead_in,
        lead_out=lead_out,
    )

    make_frame = build_frame_function(
        top_assets,
        bottom_assets,
        layout,
        top_plan=top_plan,
        bottom_plan=bottom_plan,
        background_color=background_color,
    )

    total_duration = max(top_plan.total_duration, bottom_plan.total_duration)
    video_clip = VideoClip(make_frame, duration=total_duration)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_clip.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio=False,
        preset="medium",
    )

    video_clip.close()
    return output_path


def main() -> None:
    create_film_strip_video(
        top_input_dir=SETTINGS.top_input_dir,
        output_path=SETTINGS.output_path,
        bottom_input_dir=SETTINGS.bottom_input_dir,
        target_height=SETTINGS.target_height,
        viewport_width=SETTINGS.viewport_width,
        scroll_speed=SETTINGS.scroll_speed,
        fps=SETTINGS.fps,
        lead_in=SETTINGS.lead_in,
        lead_out=SETTINGS.lead_out,
        frame_margin=SETTINGS.frame_margin,
        row_spacing=SETTINGS.row_spacing,
        stripe_height=SETTINGS.stripe_height,
        stripe_light_color=SETTINGS.stripe_light_color,
        stripe_dark_color=SETTINGS.stripe_dark_color,
        perforation_size=SETTINGS.perforation_size,
        background_color=SETTINGS.background_color,
    )


if __name__ == "__main__":
    main()
