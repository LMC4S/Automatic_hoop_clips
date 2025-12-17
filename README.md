# Basketball Shot Event Detector (ROI-Based)

(Fully AI-generated, including this README file)

Automatically extract basketball **shot attempts** from a fixed-camera video by detecting **pixel changes near the basket**.

This project avoids heavy object detection or ML. Instead, it relies on a **manually selected Region of Interest (ROI)** around the rim and detects significant visual changes when a basketball approaches the hoop — whether the shot is made or missed.

The output is a list of timestamped events and (optionally) short video clips centered on each shot.

---

## Core idea

- Camera is **fixed**
- User manually selects a **basket ROI**
- A basketball entering that area causes a **significant pixel change**
- Detect those changes → extract short clips around them

No labels, no training, no calibration.

---

## How it works (system design)

1. **ROI selection**
   - User draws a rectangle around the basket on the first frame

2. **Per-frame processing**
   - Convert ROI to grayscale
   - Apply light Gaussian blur
   - Compare against a slowly adapting background model
   - Compute a scalar “change score” per frame

3. **Event detection**
   - Smooth scores over time
   - Apply robust threshold (median + MAD)
   - Find contiguous active segments

4. **Post-processing**
   - Merge events separated by small gaps
   - Expand each event to ±5 seconds
   - Merge overlapping padded ranges

5. **Export**
   - JSON with time ranges
   - Optional MP4 clips using FFmpeg

---

## Requirements

### System
- Linux / macOS / Windows
- FFmpeg (for clip export)

### Python
- Python 3.9–3.11
- Conda strongly recommended

### Python packages
- `opencv-python`
- `numpy`

---

## Environment setup (recommended)

```bash
conda create -n basket-video python=3.10
conda activate basket-video
pip install opencv-python numpy
conda install -c conda-forge ffmpeg
```

---

## Usage

### Detect events only (JSON output)

```bash
python basket_event_finder.py input_video.mp4 --out ranges.json
```

### Detect events and export clips

```bash
python basket_event_finder.py input_video.mp4 \
  --out ranges.json \
  --export_clips_dir clips/
```

---

## Output

### `ranges.json`
```json
{
  "ranges": [
    { "start": 123.4, "end": 133.4 },
    { "start": 245.1, "end": 255.1 }
  ]
}
```

---

## Default detection parameters

```python
threshold_sigma = 6.0
merge_gap_frames = 5
pre_seconds = 5.0
post_seconds = 5.0
```

---

## License (MIT)

MIT License. See file for full text.
