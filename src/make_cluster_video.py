#!/usr/bin/env python3
"""Assemble a composite video for a given global cluster identifier."""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sqlite3
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from moviepy import CompositeAudioClip, CompositeVideoClip, ImageClip, VideoFileClip  # noqa: E402


FILENAME_PATTERN = re.compile(r"^(?P<timestamp>[^._]+)_(?P<bird0>[^._]+)_(?P<bird1>[^._]+)$")

DEFAULT_TRAIN_VIDEOS = Path("/mnt/birdconv/tb_conv_video_data")
DEFAULT_VAL_VIDEOS = Path("/mnt/birdconv/tb_conv_video_data_val")


@dataclass
class MemberRecord:
    file_path: str
    channel_index: int
    start_col: int
    end_col: int
    split: str
    distance: float | None
    source_pickle: str


@dataclass
class ClipPlan:
    video_path: Path
    start_time: float
    duration: float
    bird_label: str
    channel_index: int
    file_name: str
    block_length: float
    distance: float | None


def _with_clip_attribute(clip, attribute: str, *args, **kwargs):
    """Call MoviePy's v2 ``with_`` API, with a fallback for legacy ``set_`` methods."""
    modern = getattr(clip, f"with_{attribute}", None)
    if callable(modern):
        updated = modern(*args, **kwargs)
        return clip if updated is None else updated
    legacy = getattr(clip, f"set_{attribute}", None)
    if callable(legacy):
        updated = legacy(*args, **kwargs)
        return clip if updated is None else updated
    raise AttributeError(f"Clip does not expose with_{attribute} or set_{attribute}")


def _with_audio(clip, audio_clip):
    """Attach ``audio_clip`` to ``clip`` using either v2 or legacy APIs."""
    modern = getattr(clip, "with_audio", None)
    if callable(modern):
        return modern(audio_clip)
    legacy = getattr(clip, "set_audio", None)
    if callable(legacy):
        return legacy(audio_clip)
    clip.audio = audio_clip
    return clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-id", type=int, required=True, help="Global cluster identifier.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("cluster_registry.sqlite"),
        help="SQLite registry containing cluster membership.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "cluster_videos",
        help="Directory where the composite video will be written.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional filename for the output clip (defaults to cluster_<id>.mp4).",
    )
    parser.add_argument("--max-clips", type=int, default=16, help="Limit number of clips sampled per cluster.")
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=4.0,
        help="Maximum duration (seconds) to extract per block (defaults to 4.0).",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum duration (seconds) for an included clip. Shorter matches are skipped.",
    )
    parser.add_argument(
        "--train-videos",
        type=Path,
        default=DEFAULT_TRAIN_VIDEOS,
        help="Directory containing training MP4 clips.",
    )
    parser.add_argument(
        "--val-videos",
        type=Path,
        default=DEFAULT_VAL_VIDEOS,
        help="Directory containing validation MP4 clips.",
    )
    parser.add_argument(
        "--extra-video-root",
        type=Path,
        action="append",
        default=[],
        help="Additional directories to search for MP4 files.",
    )
    parser.add_argument(
        "--search-recursive",
        action="store_true",
        help="Recursively search video directories when matching MP4 names.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output video FPS.")
    parser.add_argument("--limit-per-channel", type=int, default=None, help="Optional per-channel clip limit.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_birds_from_stem(stem: str) -> tuple[str | None, str | None]:
    base = stem.split(".", 1)[0]
    match = FILENAME_PATTERN.match(base)
    if not match:
        return None, None
    return match.group("bird0"), match.group("bird1")


def resolve_bird_label(channel_index: int, bird0: str | None, bird1: str | None) -> str:
    if channel_index == 0:
        return bird0 or "unknown-bird0"
    if channel_index == 1:
        return bird1 or "unknown-bird1"
    if bird0 == bird1 and bird0:
        return bird0
    birds = ", ".join([bird for bird in (bird0, bird1) if bird])
    return birds or "unknown"


def fetch_members(registry_path: Path, cluster_id: int) -> list[MemberRecord]:
    if not registry_path.exists():
        raise FileNotFoundError(f"registry not found: {registry_path}")
    query = """
        SELECT file_path,
               channel_index,
               start_col,
               end_col,
               split,
               distance,
               source_pickle
        FROM members
        WHERE cluster_id = ?
        ORDER BY
            CASE WHEN distance IS NULL THEN 1 ELSE 0 END ASC,
            distance ASC,
            start_col ASC
    """
    connection = sqlite3.connect(registry_path)
    try:
        cursor = connection.cursor()
        cursor.execute(query, (int(cluster_id),))
        rows = cursor.fetchall()
    finally:
        connection.close()
    records: list[MemberRecord] = []
    for (
        file_path,
        channel_index,
        start_col,
        end_col,
        split,
        distance,
        source_pickle,
    ) in rows:
        records.append(
            MemberRecord(
                file_path=file_path,
                channel_index=int(channel_index),
                start_col=int(start_col),
                end_col=int(end_col),
                split=split or "",
                distance=float(distance) if distance is not None else None,
                source_pickle=source_pickle,
            )
        )
    return records


def candidate_video_directories(
    split: str,
    train_dir: Path,
    val_dir: Path,
    extra_dirs: Iterable[Path],
) -> list[Path]:
    split_norm = split.lower()
    directories: list[Path] = []
    if split_norm.startswith("train"):
        directories.append(train_dir)
    elif split_norm.startswith("val"):
        directories.append(val_dir)
    else:
        directories.extend([train_dir, val_dir])
    directories.extend(extra_dirs)
    return directories


def find_mp4_for_member(
    member: MemberRecord,
    train_dir: Path,
    val_dir: Path,
    extra_dirs: Iterable[Path],
    recursive: bool,
) -> Optional[Path]:
    wav_path = Path(member.file_path)
    stem = wav_path.stem
    target_name = f"{stem}.mp4"
    for directory in candidate_video_directories(member.split, train_dir, val_dir, extra_dirs):
        if not directory:
            continue
        mp4_path = directory / target_name
        if mp4_path.exists():
            return mp4_path
        if recursive and directory.exists():
            found = next(directory.rglob(target_name), None)
            if found:
                return found
    return None


@lru_cache(maxsize=512)
def load_frame_step(pickle_path: str) -> float:
    path = Path(pickle_path)
    if not path.exists():
        logging.warning("source pickle not found: %s", pickle_path)
        return 0.0
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        logging.warning("failed to load pickle %s (%s)", pickle_path, exc)
        return 0.0

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    frame_step = meta.get("frame_step")
    if frame_step:
        try:
            return float(frame_step)
        except (TypeError, ValueError):
            pass

    sr = meta.get("sr") or meta.get("sample_rate")
    hop = meta.get("hop_length") or meta.get("hop") or meta.get("step_size")
    if sr and hop:
        try:
            sr_f = float(sr)
            hop_f = float(hop)
            if sr_f > 0:
                return hop_f / sr_f
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Fallback to a default 10ms hop if nothing else is available.
    return 0.01


def compute_clip_plan(
    member: MemberRecord,
    video_path: Path,
    clip_duration: float,
    min_duration: float,
) -> Optional[ClipPlan]:
    frame_step = load_frame_step(member.source_pickle)
    if frame_step <= 0:
        return None
    block_frames = int(member.end_col - member.start_col)
    if block_frames <= 0:
        return None

    full_duration = max(0.0, block_frames * frame_step)
    if full_duration < min_duration:
        return None

    desired_duration = min(full_duration, clip_duration) if clip_duration > 0 else full_duration
    if desired_duration < min_duration:
        desired_duration = full_duration
        if desired_duration < min_duration:
            return None

    start_offset = member.start_col * frame_step
    if full_duration > desired_duration:
        start_offset += 0.5 * (full_duration - desired_duration)

    file_name = Path(member.file_path).name
    bird0, bird1 = parse_birds_from_stem(Path(member.file_path).stem)
    bird_label = resolve_bird_label(member.channel_index, bird0, bird1)

    return ClipPlan(
        video_path=video_path,
        start_time=float(start_offset),
        duration=float(desired_duration),
        bird_label=bird_label,
        channel_index=member.channel_index,
        file_name=file_name,
        block_length=float(full_duration),
        distance=member.distance,
    )


def _subclip_video(base_clip: VideoFileClip, start: float, end: float) -> VideoFileClip:
    """Return a time-sliced version of ``base_clip`` compatible with MoviePy 1.x and 2.x."""

    def _attempt(method_name: str) -> VideoFileClip | None:
        candidate = getattr(base_clip, method_name, None)
        if not callable(candidate):
            return None
        signatures = (
            ((start, end), {}),
            ((), {"start": start, "end": end}),
            ((), {"start_time": start, "end_time": end}),
            ((), {"t_start": start, "t_end": end}),
        )
        for args, kwargs in signatures:
            try:
                result = candidate(*args, **kwargs)
            except TypeError:
                continue
            if result is not None:
                return result
        return None

    for name in (
        "with_time_range",
        "with_time_slice",
        "time_slice",
        "with_subclip",
        "subclip",
        "subclipped",
    ):
        clipped = _attempt(name)
        if clipped is not None:
            return clipped

    raise AttributeError("moviepy installation lacks subclip/time_slice support")


def generate_label_array(
    text: str,
    width: int,
    *,
    height: Optional[int] = None,
    align: str = "center",
) -> np.ndarray:
    if width <= 0:
        width = 640
    if height is None:
        height = max(60, int(width * 0.06))
    dpi = 150

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black")
    fig.patch.set_alpha(0.6)
    ax = fig.add_axes([0.01, 0.0, 0.98, 1.0])
    ax.axis("off")
    align = align.lower()
    if align not in {"left", "center", "right"}:
        align = "center"
    x_coord = {"left": 0.01, "center": 0.5, "right": 0.99}[align]
    ax.text(
        x_coord,
        0.5,
        text,
        ha=align,
        va="center",
        color="white",
        fontsize=max(16, int(height * 0.45)),
        fontweight="bold",
    )
    fig.canvas.draw()
    canvas = fig.canvas
    try:
        raw = canvas.tostring_rgb()  # Matplotlib <=3.9
        buf = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
    except AttributeError:
        rgba = np.asarray(canvas.buffer_rgba())
        buf = rgba[..., :3].copy()
    plt.close(fig)
    return buf


def annotate_clip(clip: VideoFileClip, plan: ClipPlan, cluster_id: int) -> CompositeVideoClip:
    width = clip.w
    height = clip.h
    label_height = max(60, int(width * 0.06))
    position = "left" if plan.channel_index == 0 else "right"
    label_text = f"{plan.bird_label} • {plan.file_name} ({plan.block_length:.2f}s)"
    if plan.distance is not None:
        label_text += f" • d={plan.distance:.3f}"
    label_img = generate_label_array(label_text, width, height=label_height, align=position)
    label_clip = ImageClip(label_img, is_mask=False)
    label_clip = _with_clip_attribute(label_clip, "duration", clip.duration)
    label_clip = _with_clip_attribute(
        label_clip,
        "position",
        ("left" if position == "left" else "right", "bottom"),
    )

    title_height = max(60, int(width * 0.05))
    title_text = f"Cluster {cluster_id}"
    title_img = generate_label_array(title_text, width, height=title_height, align="center")
    title_clip = ImageClip(title_img, is_mask=False)
    title_clip = _with_clip_attribute(title_clip, "duration", clip.duration)
    title_clip = _with_clip_attribute(title_clip, "position", ("center", "top"))

    return CompositeVideoClip([clip, label_clip, title_clip])


def concatenate_clips(clips: list[CompositeVideoClip]) -> CompositeVideoClip:
    if not clips:
        raise ValueError("no clips to concatenate")

    arranged: list[CompositeVideoClip] = []
    audio_segments: list = []
    current_start = 0.0
    for clip in clips:
        clip_duration = float(clip.duration or 0.0)
        shifted = _with_clip_attribute(clip, "start", current_start)
        shifted = _with_clip_attribute(shifted, "end", current_start + clip_duration)
        arranged.append(shifted)
        if clip.audio:
            audio_segment = clip.audio
            audio_segment = _with_clip_attribute(audio_segment, "start", current_start)
            audio_segment = _with_clip_attribute(audio_segment, "end", current_start + clip_duration)
            if audio_segment is not None and hasattr(audio_segment, "get_frame"):
                audio_segments.append(audio_segment)
            else:
                logging.debug("audio segment dropped due to unsupported start/end setters or missing get_frame")
        current_start += clip_duration

    composite = CompositeVideoClip(arranged)
    composite = _with_clip_attribute(composite, "duration", current_start)
    filtered_audio = [
        segment
        for segment in audio_segments
        if segment is not None and hasattr(segment, "get_frame")
    ]
    if len(filtered_audio) < len(audio_segments):
        logging.debug(
            "filtered %d unusable audio segments",
            len(audio_segments) - len(filtered_audio),
        )
    audio_segments = filtered_audio
    if audio_segments:
        composite_audio = CompositeAudioClip(audio_segments)
        composite_audio = _with_clip_attribute(composite_audio, "duration", current_start)
        composite = _with_audio(composite, composite_audio)
    return composite


def assemble_video(
    plans: list[ClipPlan],
    output_path: Path,
    cluster_id: int,
    fps: Optional[float],
    min_duration: float,
) -> None:
    temp_clips: list[VideoFileClip] = []
    annotated_clips: list[CompositeVideoClip] = []

    try:
        for plan in plans:
            try:
                base_clip = VideoFileClip(str(plan.video_path))
            except OSError as exc:
                logging.warning("failed to open %s (%s)", plan.video_path, exc)
                continue

            available = float(base_clip.duration or 0.0)
            if available <= 0:
                base_clip.close()
                continue
            clip_start = min(max(plan.start_time, 0.0), max(0.0, available - 1e-3))
            clip_end = min(available, clip_start + plan.duration)
            if clip_end - clip_start < min_duration:
                extension = min_duration - (clip_end - clip_start)
                clip_start = max(0.0, clip_start - extension / 2)
                clip_end = min(available, clip_end + extension / 2)
            if clip_end - clip_start < min_duration:
                base_clip.close()
                continue

            subclip = _subclip_video(base_clip, clip_start, clip_end)
            base_clip.close()
            temp_clips.append(subclip)
            annotated = annotate_clip(subclip, plan, cluster_id)
            annotated_clips.append(annotated)

        if not annotated_clips:
            raise RuntimeError("no usable clips were gathered for the requested cluster.")

        final = concatenate_clips(annotated_clips)
        logging.info("writing composite video (%s)", output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_kwargs = {"codec": "libx264", "audio_codec": "aac"}
        if fps:
            write_kwargs["fps"] = fps
        final.write_videofile(str(output_path), **write_kwargs)
        final.close()
    finally:
        for clip in annotated_clips:
            clip.close()
        for clip in temp_clips:
            clip.close()


def select_plans(
    members: list[MemberRecord],
    args: argparse.Namespace,
) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    seen: set[tuple[str, int, int, int]] = set()
    per_channel_counter: dict[int, int] = {}

    for member in members:
        key = (member.file_path, member.channel_index, member.start_col, member.end_col)
        if key in seen:
            continue
        seen.add(key)

        channel_count = per_channel_counter.get(member.channel_index, 0)
        if args.limit_per_channel is not None and channel_count >= args.limit_per_channel:
            continue

        mp4_path = find_mp4_for_member(
            member,
            args.train_videos,
            args.val_videos,
            args.extra_video_root,
            args.search_recursive,
        )
        if not mp4_path:
            logging.debug("no mp4 found for %s (%s)", member.file_path, member.split)
            continue

        plan = compute_clip_plan(
            member,
            mp4_path,
            args.clip_duration,
            args.min_duration,
        )
        if not plan:
            continue

        plans.append(plan)
        per_channel_counter[member.channel_index] = channel_count + 1
        if len(plans) >= args.max_clips:
            break
    return plans


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        members = fetch_members(args.registry, args.cluster_id)
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to fetch cluster members: %s", exc)
        sys.exit(1)

    if not members:
        logging.error("cluster %s has no registered members.", args.cluster_id)
        sys.exit(1)

    plans = select_plans(members, args)
    if not plans:
        logging.error("no usable video clips found for cluster %s.", args.cluster_id)
        sys.exit(1)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = args.filename or f"cluster_{args.cluster_id}.mp4"
    output_path = output_dir / output_filename

    try:
        assemble_video(plans, output_path, args.cluster_id, args.fps, args.min_duration)
    except Exception as exc:  # noqa: BLE001
        logging.error("failed to assemble video: %s", exc)
        sys.exit(1)
    logging.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
