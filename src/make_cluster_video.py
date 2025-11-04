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
import matplotlib.font_manager as font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from moviepy import CompositeAudioClip, CompositeVideoClip, ImageClip, VideoFileClip  # noqa: E402


FILENAME_PATTERN = re.compile(r"^(?P<timestamp>[^._]+)_(?P<bird0>[^._]+)_(?P<bird1>[^._]+)$")

DEFAULT_TRAIN_VIDEOS = Path("/mnt/birdconv/tb_conv_video_data")
DEFAULT_VAL_VIDEOS = Path("/mnt/birdconv/tb_conv_video_data_val")


@dataclass
class MemberRecord:
    file_path: str
    channel_index: int
    start_col: float
    end_col: float
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


_COLUMN_PRECISION = 6


def _normalize_col(value: float) -> float:
    return round(float(value), _COLUMN_PRECISION)


def _describe_clip(clip) -> str:
    """Return a concise string describing a MoviePy clip instance."""
    if clip is None:
        return "None"
    attrs = []
    for name in ("start", "end", "duration", "fps"):
        value = getattr(clip, name, None)
        if value is not None:
            attrs.append(f"{name}={value!r}")
    has_audio = getattr(clip, "audio", None) is not None
    attrs.append(f"audio={has_audio}")
    return f"{clip.__class__.__name__}({', '.join(attrs)})"


def _with_clip_attribute(clip, attribute: str, *args, **kwargs):
    """Call MoviePy's v2 ``with_`` API, with a fallback for legacy ``set_`` methods."""
    modern = getattr(clip, f"with_{attribute}", None)
    if callable(modern):
        logging.debug("calling %s.with_%s%s", clip.__class__.__name__, attribute, (args, kwargs))
        updated = modern(*args, **kwargs)
        if updated is None:
            logging.debug("%s.with_%s returned None; reusing original clip", clip.__class__.__name__, attribute)
            return clip
        return updated
    legacy = getattr(clip, f"set_{attribute}", None)
    if callable(legacy):
        logging.debug("calling %s.set_%s%s", clip.__class__.__name__, attribute, (args, kwargs))
        updated = legacy(*args, **kwargs)
        if updated is None:
            logging.debug("%s.set_%s returned None; reusing original clip", clip.__class__.__name__, attribute)
            return clip
        return updated
    raise AttributeError(f"Clip does not expose with_{attribute} or set_{attribute}")


def _with_audio(clip, audio_clip):
    """Attach ``audio_clip`` to ``clip`` using either v2 or legacy APIs."""
    modern = getattr(clip, "with_audio", None)
    if callable(modern):
        logging.debug("attaching audio via with_audio: %s <- %s", _describe_clip(clip), _describe_clip(audio_clip))
        return modern(audio_clip)
    legacy = getattr(clip, "set_audio", None)
    if callable(legacy):
        logging.debug("attaching audio via set_audio: %s <- %s", _describe_clip(clip), _describe_clip(audio_clip))
        return legacy(audio_clip)
    clip.audio = audio_clip
    logging.debug("attaching audio via attribute assignment: %s <- %s", _describe_clip(clip), _describe_clip(audio_clip))
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
                start_col=float(start_col),
                end_col=float(end_col),
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
        logging.debug(
            "skipping %s: invalid frame_step=%s (source=%s)",
            member.file_path,
            frame_step,
            member.source_pickle,
        )
        return None
    block_cols = max(0.0, float(member.end_col) - float(member.start_col))
    if block_cols <= 0:
        logging.debug(
            "skipping %s: non-positive block frame count (%s -> %s)",
            member.file_path,
            member.start_col,
            member.end_col,
        )
        return None

    full_duration = max(0.0, block_cols * frame_step)
    if full_duration < min_duration:
        logging.debug(
            "skipping %s: block duration %.3fs < min_duration %.3fs",
            member.file_path,
            full_duration,
            min_duration,
        )
        return None

    desired_duration = min(full_duration, clip_duration) if clip_duration > 0 else full_duration
    if desired_duration < min_duration:
        desired_duration = full_duration
        if desired_duration < min_duration:
            logging.debug(
                "skipping %s: desired duration %.3fs < min_duration %.3fs after adjustment",
                member.file_path,
                desired_duration,
                min_duration,
            )
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


_FONT_CACHE: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def _line_height(font: ImageFont.ImageFont) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except Exception:  # noqa: BLE001
        bbox = font.getbbox("Hg")
        return max(1, bbox[3] - bbox[1])


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    key = (size, bold)
    if key not in _FONT_CACHE:
        family = "DejaVu Sans Bold" if bold else "DejaVu Sans"
        path = font_manager.findfont(family, fallback_to_default=True)
        try:
            font = ImageFont.truetype(path, size=size)
        except (OSError, ValueError):
            font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not candidate:
            continue
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        if draw.textlength(word, font=font) <= max_width:
            current = word
        else:
            chunk = ""
            for char in word:
                trial = chunk + char
                if draw.textlength(trial, font=font) > max_width and chunk:
                    lines.append(chunk)
                    chunk = char
                else:
                    chunk = trial
            current = chunk
    if current:
        lines.append(current)
    return lines


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
    width = int(width)
    height = int(height)
    margin = max(int(min(width, height) * 0.1), 12)
    max_text_width = max(1, width - 2 * margin)

    image = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)

    align = align.lower()
    if align not in {"left", "center", "right"}:
        align = "center"
    lines = text.splitlines() or [text]
    header = lines[0].strip() if lines else ""
    body = " ".join(line.strip() for line in lines[1:] if line.strip())

    y = margin
    remaining_height = height - margin

    if header:
        header_limit = max(int(remaining_height * 0.45), 18)
        header_font = _load_font(header_limit, bold=True)
        for size in range(header_limit, 12, -1):
            header_font = _load_font(size, bold=True)
            if draw.textlength(header, font=header_font) <= max_text_width:
                break
        header_height = _line_height(header_font)
        if header_height > remaining_height:
            header_height = remaining_height
        header_width = draw.textlength(header, font=header_font)
        if align == "left":
            x = margin
        elif align == "right":
            x = max(margin, width - margin - header_width)
        else:
            x = (width - header_width) / 2
        draw.text((x, y), header, font=header_font, fill=(255, 255, 255))
        y += header_height + max(4, int(header_font.size * 0.25))
        remaining_height = max(0, height - y - margin)

    if body and remaining_height > 0:
        max_body_size = max(int(remaining_height * 0.35), 14)
        body_font = _load_font(max_body_size, bold=False)
        body_lines = _wrap_text(draw, body, body_font, max_text_width)
        for size in range(max_body_size, 10, -1):
            body_font = _load_font(size, bold=False)
            body_lines = _wrap_text(draw, body, body_font, max_text_width)
            line_height = _line_height(body_font)
            spacing = max(4, int(body_font.size * 0.25))
            body_height = len(body_lines) * line_height + max(0, len(body_lines) - 1) * spacing
            if body_height <= remaining_height:
                break
        spacing = max(4, int(body_font.size * 0.25))
        for line in body_lines:
            line_width = draw.textlength(line, font=body_font)
            if align == "left":
                x = margin
            elif align == "right":
                x = max(margin, width - margin - line_width)
            else:
                x = (width - line_width) / 2
            draw.text((x, y), line, font=body_font, fill=(220, 220, 220))
            y += _line_height(body_font) + spacing

    return np.asarray(image)


def annotate_clip(clip: VideoFileClip, plan: ClipPlan, cluster_id: int) -> CompositeVideoClip:
    width = clip.w
    height = clip.h
    label_width = max(int(width * 0.45), 320)
    label_height = max(90, int(label_width * 0.32))
    position = "left" if plan.channel_index == 0 else "right"
    metadata_parts = [f"{plan.file_name} ({plan.block_length:.2f}s)"]
    if plan.distance is not None:
        metadata_parts.append(f"d={plan.distance:.3f}")
    label_text = f"{plan.bird_label}\n" + " • ".join(metadata_parts)
    label_img = generate_label_array(label_text, label_width, height=label_height, align=position)
    label_clip = ImageClip(label_img, is_mask=False)
    label_clip = _with_clip_attribute(label_clip, "duration", clip.duration)
    margin = max(int(width * 0.02), 20)
    pos_x = margin if position == "left" else max(width - label_width - margin, margin)
    pos_y = max(height - label_height - margin, margin)
    label_clip = _with_clip_attribute(label_clip, "position", (pos_x, pos_y))

    title_height = max(60, int(width * 0.05))
    title_text = f"Cluster {cluster_id}"
    title_img = generate_label_array(title_text, width, height=title_height, align="center")
    title_clip = ImageClip(title_img, is_mask=False)
    title_clip = _with_clip_attribute(title_clip, "duration", clip.duration)
    title_clip = _with_clip_attribute(title_clip, "position", ("center", "top"))

    composite = CompositeVideoClip([clip, label_clip, title_clip])
    base_audio = getattr(clip, "audio", None)
    if base_audio is not None:
        logging.debug(
            "propagating audio for clip %s: %s -> %s",
            plan.file_name,
            _describe_clip(base_audio),
            _describe_clip(composite),
        )
        composite = _with_audio(composite, base_audio)
    else:
        logging.debug("clip %s has no audio to propagate", plan.file_name)
    return composite


def concatenate_clips(clips: list[CompositeVideoClip]) -> CompositeVideoClip:
    if not clips:
        raise ValueError("no clips to concatenate")

    arranged: list[CompositeVideoClip] = []
    audio_segments: list = []
    current_start = 0.0
    for clip in clips:
        clip_duration = float(clip.duration or 0.0)
        logging.debug(
            "processing clip %s duration=%s audio=%s",
            _describe_clip(clip),
            clip_duration,
            hasattr(clip, "audio") and clip.audio is not None,
        )
        shifted = _with_clip_attribute(clip, "start", current_start)
        shifted = _with_clip_attribute(shifted, "end", current_start + clip_duration)
        arranged.append(shifted)
        if clip.audio:
            audio_segment = clip.audio
            logging.debug("  original audio segment: %s", _describe_clip(audio_segment))
            audio_segment = _with_clip_attribute(audio_segment, "start", current_start)
            audio_segment = _with_clip_attribute(audio_segment, "end", current_start + clip_duration)
            if audio_segment is not None and hasattr(audio_segment, "get_frame"):
                logging.debug("  prepared audio segment: %s", _describe_clip(audio_segment))
                audio_segments.append(audio_segment)
            else:
                logging.debug("audio segment dropped due to unsupported start/end setters or missing get_frame")
        current_start += clip_duration

    composite = CompositeVideoClip(arranged)
    composite = _with_clip_attribute(composite, "duration", current_start)
    logging.debug("built video composite: %s", _describe_clip(composite))
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
        logging.debug("final audio segments: %s", [ _describe_clip(seg) for seg in audio_segments ])
        composite_audio = CompositeAudioClip(audio_segments)
        composite_audio = _with_clip_attribute(composite_audio, "duration", current_start)
        composite = _with_audio(composite, composite_audio)
        logging.debug("attached composite audio: %s", _describe_clip(composite.audio))
    else:
        logging.debug("no usable audio segments, returning silent composite")
    return composite


def assemble_video(
    plans: list[ClipPlan],
    output_path: Path,
    cluster_id: int,
    fps: Optional[float],
    min_duration: float,
) -> None:
    opened_clips: list[VideoFileClip] = []
    annotated_clips: list[CompositeVideoClip] = []

    try:
        for plan in plans:
            try:
                base_clip = VideoFileClip(str(plan.video_path))
            except OSError as exc:
                logging.warning("failed to open %s (%s)", plan.video_path, exc)
                continue

            opened_clips.append(base_clip)
            available = float(base_clip.duration or 0.0)
            if available <= 0:
                continue
            clip_start = min(max(plan.start_time, 0.0), max(0.0, available - 1e-3))
            clip_end = min(available, clip_start + plan.duration)
            if clip_end - clip_start < min_duration:
                extension = min_duration - (clip_end - clip_start)
                clip_start = max(0.0, clip_start - extension / 2)
                clip_end = min(available, clip_end + extension / 2)
            if clip_end - clip_start < min_duration:
                continue

            subclip = _subclip_video(base_clip, clip_start, clip_end)
            opened_clips.append(subclip)
            annotated = annotate_clip(subclip, plan, cluster_id)
            logging.debug(
                "annotated clip for %s: %s (audio=%s)",
                plan.file_name,
                _describe_clip(annotated),
                hasattr(annotated, "audio") and annotated.audio is not None,
            )
            annotated_clips.append(annotated)

        if not annotated_clips:
            raise RuntimeError("no usable clips were gathered for the requested cluster.")

        annotated_clips.sort(key=lambda clip: float(clip.duration or 0.0))
        logging.debug(
            "sorted clip durations: %s",
            [float(clip.duration or 0.0) for clip in annotated_clips],
        )

        final = concatenate_clips(annotated_clips)
        audio_clip = getattr(final, "audio", None)
        if audio_clip is None:
            logging.debug("final composite has no audio")
        else:
            logging.debug(
                "final composite audio summary: %s containing %d segments",
                _describe_clip(audio_clip),
                len(getattr(audio_clip, "clips", []) or []),
            )
            segments = getattr(audio_clip, "clips", []) or []
            for idx, segment in enumerate(segments):
                logging.debug(
                    "audio clip %d: %s",
                    idx,
                    _describe_clip(segment),
                )
                probe_time = max(0.0, min(0.001, float(getattr(segment, "duration", 0.0)) / 2))
                try:
                    sample = segment.get_frame(probe_time)
                except Exception as exc:  # noqa: BLE001
                    logging.debug("audio clip %d get_frame failed at %s: %s", idx, probe_time, exc)
                else:
                    if sample is None:
                        logging.debug("audio clip %d get_frame(%s) returned None", idx, probe_time)
                    else:
                        logging.debug(
                            "audio clip %d get_frame(%s) sample type=%s shape=%s",
                            idx,
                            probe_time,
                            type(sample),
                            getattr(sample, "shape", None),
                        )

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
        for clip in opened_clips:
            clip.close()


def select_plans(
    members: list[MemberRecord],
    args: argparse.Namespace,
) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    seen: set[tuple[str, int, float, float]] = set()
    per_channel_counter: dict[int, int] = {}

    for member in members:
        key = (
            member.file_path,
            member.channel_index,
            _normalize_col(member.start_col),
            _normalize_col(member.end_col),
        )
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