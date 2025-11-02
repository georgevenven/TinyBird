#!/usr/bin/env bash
#
# Generate per-segment MP4 clips aligned with clustered WAV files.
# Usage: run from anywhere; paths are hard-coded to the project layout.

set -uo pipefail

readonly TRAIN_WAV_DIR=${TRAIN_WAV_DIR:-/mnt/birdconv/tb_conv_data}
readonly VAL_WAV_DIR=${VAL_WAV_DIR:-/mnt/birdconv/tb_conv_data_val}
readonly TRAIN_OUT_DIR=${TRAIN_OUT_DIR:-/mnt/birdconv/tb_conv_video_data}
readonly VAL_OUT_DIR=${VAL_OUT_DIR:-/mnt/birdconv/tb_conv_video_data_val}
readonly SOURCE_VIDEO_ROOT=${SOURCE_VIDEO_ROOT:-/mnt/birdconv/data}
readonly FFMPEG_BIN=${FFMPEG_BIN:-$(command -v ffmpeg)}

readonly WAV_DIRS=(${TRAIN_WAV_DIR} ${VAL_WAV_DIR})
readonly MP4_DIRS=(${TRAIN_OUT_DIR} ${VAL_OUT_DIR})

declare -A MP4_CACHE
declare -a MISSING_VIDEO_NAMES=()

find_source_video() {
    local timestamp="$1"
    local bird0="$2"
    local bird1="$3"

    local key="${timestamp}-${bird0}-${bird1}"
    local rev_key="${timestamp}-${bird1}-${bird0}"

    if [[ -n ${MP4_CACHE[$key]+x} ]]; then
        printf '%s\n' "${MP4_CACHE[$key]}"
        return 0
    fi

    local path
    path=$(find "$SOURCE_VIDEO_ROOT" -type f -name "${key}.mp4" -print -quit 2>/dev/null || true)
    if [[ -n "$path" ]]; then
        MP4_CACHE[$key]="normal|$path"
        MP4_CACHE[$rev_key]="swap|$path"
        printf '%s\n' "${MP4_CACHE[$key]}"
        return 0
    fi

    path=$(find "$SOURCE_VIDEO_ROOT" -type f -name "${rev_key}.mp4" -print -quit 2>/dev/null || true)
    if [[ -n "$path" ]]; then
        MP4_CACHE[$key]="swap|$path"
        MP4_CACHE[$rev_key]="normal|$path"
        printf '%s\n' "${MP4_CACHE[$key]}"
        return 0
    fi

    return 1
}

process_wav() {
    local wav="$1"
    local out_dir="$2"

    local filename
    filename="$(basename "$wav")"
    local base="${filename%.wav}"

    local timestamp="${base%%_*}"
    local remainder="${base#*_}"
    local bird0="${remainder%%_*}"
    remainder="${remainder#*_}"
    local bird1="${remainder%%.*}"
    remainder="${remainder#*.}"
    local start="${remainder%%_*}"
    local length="${remainder#*_}"

    local key1="${timestamp}-${bird0}-${bird1}.mp4"
    local key2="${timestamp}-${bird1}-${bird0}.mp4"

    local result
    if ! result=$(find_source_video "$timestamp" "$bird0" "$bird1"); then
        echo "  [MISS] source video not found for $filename (tried ${key1} / ${key2})"
        MISSING_VIDEO_NAMES+=("${key1};${key2}")
        return 1
    fi

    IFS='|' read -r orientation source_path <<<"$result"

    local output_path="${out_dir}/${base}.mp4"
    if [[ -f "$output_path" ]]; then
        echo "  [SKIP] $output_path already exists"
        return 3
    fi

    mkdir -p "$out_dir"

    local ffmpeg_cmd=("$FFMPEG_BIN" -hide_banner -loglevel error -ss "$start" -i "$source_path" -t "$length")

    if [[ "$orientation" == "swap" ]]; then
        ffmpeg_cmd+=(
            -filter_complex
            "[0:v]split[va][vb];[va]crop=iw/2:ih:0:0[left];[vb]crop=iw/2:ih:iw/2:0[right];[right][left]hstack=inputs=2[vout];[0:a]pan=stereo|c0=c1|c1=c0[aout]"
            -map "[vout]" -map "[aout]" -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k
        )
    else
        ffmpeg_cmd+=(-c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 192k)
    fi

    ffmpeg_cmd+=("$output_path")

    echo "  [GEN] ${filename} -> $(basename "$output_path") (${orientation})"
    if ! "${ffmpeg_cmd[@]}"; then
        echo "  [FAIL] ffmpeg failed for $filename"
        return 2
    fi

    return 0
}

main() {
    local total_processed=0
    local total_skipped=0
    local total_missing=0
    local total_failed=0

    if [[ -z "$FFMPEG_BIN" ]]; then
        echo "[ERROR] ffmpeg binary not found. Set FFMPEG_BIN or install ffmpeg." >&2
        return 1
    fi

    echo "Using ffmpeg: $FFMPEG_BIN"
    echo "Source video root: $SOURCE_VIDEO_ROOT"
    echo "Train WAV dir: ${WAV_DIRS[0]} -> ${MP4_DIRS[0]}"
    echo "Val   WAV dir: ${WAV_DIRS[1]} -> ${MP4_DIRS[1]}"

    for idx in "${!WAV_DIRS[@]}"; do
        local wav_dir="${WAV_DIRS[$idx]}"
        local out_dir="${MP4_DIRS[$idx]}"
        echo "Processing directory: $wav_dir"
        if [[ ! -d "$wav_dir" ]]; then
            echo "  [WARN] directory not found: $wav_dir"
            continue
        fi
        local count
        count=$(find "$wav_dir" -type f -name "*.wav" | wc -l | awk '{print $1}')
        if (( count == 0 )); then
            echo "  [INFO] no WAV files found in $wav_dir"
            continue
        fi
        local idx=0
        while IFS= read -r -d '' wav; do
            idx=$((idx + 1))
            printf "[%d/%d] %s\n" "$idx" "$count" "$(basename "$wav")"
            if process_wav "$wav" "$out_dir"; then
                ((total_processed++))
            else
                status=$?
                case $status in
                    1) ((total_missing++)) ;;
                    2) ((total_failed++)) ;;
                    3) ((total_skipped++)) ;;
                    *) ;;
                esac
            fi
        done < <(find "$wav_dir" -type f -name "*.wav" -print0 | sort -z)
    done

    echo "Summary:"
    echo "  Generated : $total_processed"
    echo "  Missing   : $total_missing"
    echo "  Failed    : $total_failed"
    echo "  Skipped   : $total_skipped"

    if (( total_missing > 0 )); then
        echo "Missing source MP4 candidates (searched under ${SOURCE_VIDEO_ROOT}):"
        for entry in "${MISSING_VIDEO_NAMES[@]}"; do
            IFS=';' read -r primary secondary <<<"$entry"
            printf "  %s or %s\n" "$primary" "$secondary"
        done
    fi
}

main "$@"
