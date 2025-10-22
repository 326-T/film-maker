"""Utilities for generating a film-strip style scrolling video from still images.

The script reads every supported image from an input directory, stitches the
frames into a single horizontal strip, and creates a MoviePy ``VideoClip`` that
scrolls through the strip sideways – mimicking a film reel moving past the
camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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

    input_dir: Path = Path("input")
    output_path: Path = Path("output/film_strip.mp4")
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


def create_film_strip_video(
    input_dir: Path,
    output_path: Path,
    *,
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

    The input images are stitched into a single strip that is shown twice: the
    upper row scrolls left-to-right, while the lower row scrolls right-to-left.
    Each row is framed by a perforated stripe pattern to evoke the look of
    analogue film.
    """

    if target_height <= 0:
        msg = "target_height must be a positive integer"
        raise ValueError(msg)
    if scroll_speed <= 0:
        msg = "scroll_speed must be greater than 0"
        raise ValueError(msg)
    if fps <= 0:
        msg = "fps must be greater than 0"
        raise ValueError(msg)
    if lead_in < 0 or lead_out < 0:
        msg = "lead_in and lead_out must be non-negative"
        raise ValueError(msg)
    if frame_margin < 0 or row_spacing < 0:
        msg = "Layout spacing must be non-negative"
        raise ValueError(msg)
    if stripe_height <= 0:
        msg = "stripe_height must be positive"
        raise ValueError(msg)
    if perforation_size <= 0:
        msg = "perforation_size must be positive"
        raise ValueError(msg)

    usable_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
        ),
        key=natural_key,
    )

    if not usable_paths:
        msg = f"No supported images found in {input_dir}"
        raise FileNotFoundError(msg)

    frames, strip_width = load_resized_frames(usable_paths, target_height)
    strip_frame = build_strip(frames)

    viewport_width = viewport_width or strip_width
    viewport_width = min(viewport_width, strip_width)

    scroll_distance = max(strip_width - viewport_width, 0)
    scroll_duration = scroll_distance / scroll_speed if scroll_distance else 0.0
    total_duration = lead_in + scroll_duration + lead_out

    row_height = strip_frame.shape[0]
    row_block_height = row_height + (stripe_height * 2)
    frame_height = frame_margin * 2 + row_block_height * 2 + row_spacing

    perforation_strip = create_perforation_band(
        strip_width,
        stripe_height,
        band_color=stripe_dark_color,
        perforation_color=stripe_light_color,
        square_size=perforation_size,
    )

    background = np.full(
        (frame_height, viewport_width, 3),
        fill_value=background_color,
        dtype=np.uint8,
    )

    top_row_top = frame_margin
    bottom_row_top = frame_margin + row_block_height + row_spacing
    top_image_top = top_row_top + stripe_height
    bottom_image_top = bottom_row_top + stripe_height

    def compute_offset(t: float, *, reverse: bool) -> int:
        if scroll_distance == 0:
            return 0

        if t <= lead_in:
            progress = 0.0
        elif t >= lead_in + scroll_duration:
            progress = float(scroll_distance)
        else:
            progress = (t - lead_in) * scroll_speed

        progress = max(0.0, min(progress, float(scroll_distance)))

        offset = float(scroll_distance) - progress if reverse else progress
        offset_int = int(round(offset))
        return max(0, min(offset_int, scroll_distance))

    def top_row_frame(t: float) -> np.ndarray:
        offset = compute_offset(t, reverse=False)
        return strip_frame[:, offset : offset + viewport_width, :].copy()

    def bottom_row_frame(t: float) -> np.ndarray:
        offset = compute_offset(t, reverse=True)
        return strip_frame[:, offset : offset + viewport_width, :].copy()

    def band_frame(t: float, *, reverse: bool) -> np.ndarray:
        offset = compute_offset(t, reverse=reverse)
        return perforation_strip[:, offset : offset + viewport_width, :].copy()

    def make_frame(t: float) -> np.ndarray:
        frame = background.copy()

        top_strip = top_row_frame(t)
        bottom_strip = bottom_row_frame(t)

        top_band = band_frame(t, reverse=False)
        bottom_band = band_frame(t, reverse=True)

        frame[top_row_top : top_row_top + stripe_height, :, :] = top_band
        frame[top_image_top : top_image_top + row_height, :, :] = top_strip
        frame[
            top_image_top + row_height : top_image_top + row_height + stripe_height,
            :,
            :,
        ] = top_band

        frame[bottom_row_top : bottom_row_top + stripe_height, :, :] = bottom_band
        frame[bottom_image_top : bottom_image_top + row_height, :, :] = bottom_strip
        frame[
            bottom_image_top + row_height : bottom_image_top
            + row_height
            + stripe_height,
            :,
            :,
        ] = bottom_band

        return frame

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
        input_dir=SETTINGS.input_dir,
        output_path=SETTINGS.output_path,
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
