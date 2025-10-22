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
    target_height: int = 720
    viewport_width: int = 1920
    scroll_speed: float = 240.0
    fps: int = 30
    lead_in: float = 0.75
    lead_out: float = 0.75


SETTINGS = FilmStripSettings()


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
        clip = ImageClip(str(path)).resize(height=target_height)
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
) -> Path:
    """Generate a horizontally scrolling film-strip style video.

    Args:
        input_dir: Directory containing source images.
        output_path: Output path for the generated video file.
        target_height: Height (in pixels) to scale each source image.
        viewport_width: Width of the cropped window that moves across the
            stitched strip. ``None`` defaults to the strip width (no scroll).
        scroll_speed: Pixels-per-second that the strip moves under the viewport.
        fps: Frames per second for the exported video.
        lead_in: Seconds to hold on the first frame before scrolling starts.
        lead_out: Seconds to hold on the final frame after scrolling completes.
    Returns:
        The output path once the video has been written.
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

    strip_height = strip_frame.shape[0]

    def make_frame(t: float) -> np.ndarray:
        if scroll_distance == 0:
            return strip_frame[:, :viewport_width]

        if t <= lead_in:
            offset = 0
        elif t >= lead_in + scroll_duration:
            offset = scroll_distance
        else:
            offset = int(round((t - lead_in) * scroll_speed))
            offset = min(offset, scroll_distance)

        return strip_frame[:, offset : offset + viewport_width]

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
    )


if __name__ == "__main__":
    main()
