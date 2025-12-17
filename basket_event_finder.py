import cv2
import numpy as np
import json
import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Params:
    # Detection
    downscale_width: int = 960          # speed-up; set None to disable
    blur_ksize: int = 7                 # odd number; 0 disables blur
    bg_alpha: float = 0.02              # background update rate (EMA)
    score_smooth_window: int = 9        # moving average window (odd is nice)

    # Thresholding
    threshold_sigma: float = 6.0        # threshold = median + sigma * MAD
    min_active_frames: int = 6          # discard very short blips
    merge_gap_frames: int = 5         # merge events if gap <= this many frames

    # Clip padding
    pre_seconds: float = 5.0
    post_seconds: float = 2.0


def robust_threshold(scores: np.ndarray, sigma: float) -> float:
    """median + sigma * MAD (robust to spikes)."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med)) + 1e-9
    # MAD -> approx std for normal: 1.4826 * MAD
    robust_std = 1.4826 * mad
    return float(med + sigma * robust_std)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


def select_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns ROI as (x, y, w, h)."""
    r = cv2.selectROI("Select basket ROI (ENTER to confirm)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select basket ROI (ENTER to confirm)")
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        raise ValueError("ROI selection cancelled or invalid.")
    return x, y, w, h


def read_first_frame(video_path: str, downscale_width: int | None) -> Tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read first frame.")

    frame = maybe_downscale(frame, downscale_width)
    return frame, float(fps), n_frames


def maybe_downscale(frame: np.ndarray, downscale_width: int | None) -> np.ndarray:
    if downscale_width is None:
        return frame
    h, w = frame.shape[:2]
    if w <= downscale_width:
        return frame
    scale = downscale_width / w
    new_size = (downscale_width, int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def compute_scores(video_path: str, roi: Tuple[int, int, int, int], params: Params) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    x, y, w, h = roi
    scores = []

    bg = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = maybe_downscale(frame, params.downscale_width)

        # ROI crop
        roi_frame = frame[y:y+h, x:x+w]
        if roi_frame.size == 0:
            cap.release()
            raise ValueError("ROI is out of bounds after downscaling. Re-select ROI or disable downscale.")

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if params.blur_ksize and params.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (params.blur_ksize, params.blur_ksize), 0)

        # Init background as first ROI
        if bg is None:
            bg = gray.copy()

        # Score = mean abs difference from background
        diff = np.abs(gray - bg)
        score = float(np.mean(diff))
        scores.append(score)

        # Update background slowly (EMA) so it adapts to lighting drift
        bg = (1.0 - params.bg_alpha) * bg + params.bg_alpha * gray

        frame_idx += 1

    cap.release()
    return np.array(scores, dtype=np.float32), fps


def detect_segments(scores: np.ndarray, fps: float, params: Params) -> Tuple[List[Tuple[float, float]], dict]:
    sm = moving_average(scores, params.score_smooth_window)
    thr = robust_threshold(sm, params.threshold_sigma)

    active = sm >= thr

    # Convert active boolean array to (start_frame, end_frame) inclusive
    segments = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            in_seg = True
            start = i
        elif not a and in_seg:
            in_seg = False
            end = i - 1
            segments.append((start, end))
    if in_seg:
        segments.append((start, len(active) - 1))

    # Drop tiny segments
    segments = [(s, e) for (s, e) in segments if (e - s + 1) >= params.min_active_frames]

    # Merge close segments
    merged = []
    for s, e in segments:
        if not merged:
            merged.append([s, e])
            continue
        ps, pe = merged[-1]
        if s - pe <= params.merge_gap_frames:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    # Expand by pre/post seconds and convert to time ranges
    ranges = []
    for s, e in merged:
        t0 = max(0.0, (s / fps) - params.pre_seconds)
        t1 = (e / fps) + params.post_seconds
        ranges.append((t0, t1))

    # Merge overlaps after padding
    ranges.sort()
    final = []
    for t0, t1 in ranges:
        if not final or t0 > final[-1][1]:
            final.append([t0, t1])
        else:
            final[-1][1] = max(final[-1][1], t1)

    diagnostics = {
        "fps": fps,
        "threshold": thr,
        "num_frames": int(len(scores)),
        "num_segments_raw": int(len(segments)),
        "num_segments_final": int(len(final)),
        "score_median": float(np.median(sm)),
        "score_max": float(np.max(sm)),
    }

    return [(float(a), float(b)) for a, b in final], diagnostics


def save_ranges_json(ranges: List[Tuple[float, float]], diagnostics: dict, out_path: str):
    payload = {"ranges": [{"start": a, "end": b} for a, b in ranges], "diagnostics": diagnostics}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def export_clips_ffmpeg(video_path: str, ranges: List[Tuple[float, float]], out_dir: str):
    """
    Requires ffmpeg installed on your system.
    Uses stream copy when possible (fast). If you see A/V sync issues, switch to re-encode.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    for i, (t0, t1) in enumerate(ranges, start=1):
        out_path = os.path.join(out_dir, f"{base}_clip_{i:03d}.mp4")
        duration = max(0.01, t1 - t0)

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{t0:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            out_path,
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Find shot-like events near a manually selected basket ROI.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--out", default="ranges.json", help="Output JSON path")
    parser.add_argument("--export_clips_dir", default=None, help="If set, export mp4 clips to this folder (requires ffmpeg)")
    parser.add_argument("--no_downscale", action="store_true", help="Disable downscaling")
    args = parser.parse_args()

    params = Params()
    if args.no_downscale:
        params.downscale_width = None

    first, fps, _ = read_first_frame(args.video, params.downscale_width)
    roi = select_roi(first)

    scores, fps = compute_scores(args.video, roi, params)
    ranges, diag = detect_segments(scores, fps, params)
    save_ranges_json(ranges, diag, args.out)

    print(f"Wrote {len(ranges)} ranges to {args.out}")
    print("Diagnostics:", diag)

    if args.export_clips_dir:
        export_clips_ffmpeg(args.video, ranges, args.export_clips_dir)
        print(f"Exported clips to: {args.export_clips_dir}")


if __name__ == "__main__":
    main()