# LLaVA-OneVision-2 Codec-Backend SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing StreamingVLM SFT pipeline for LLaVA-OneVision-2 (already running on the frames backend per `docs/superpowers/plans/2026-05-27-llava-ov2-sft.md`) with a second video backend that uses `cv-preinfer` codec canvases, while preserving the multi-round chunked-streaming training semantics (each canvas slice goes into one chunk-round of the dialog).

**Architecture:**
- The codec backend is treated as an alternative *frame source* for `LMMDataset.preprocess_conversation_stream`: instead of decord-decoding the full clip and uniformly sampling, we call `process_codec_video(video_url, cfg)` (on-the-fly; its built-in disk cache makes the first call slow and subsequent calls instant). The result is N canvases + per-patch `(t, h, w)` source positions.
- We slice those canvases by their per-canvas time range into chunks, exactly mirroring the existing frames-backend chunking logic, but with canvases (PIL images, irregular cadence) standing in for raw frames (uint8 tensor stacks, uniform cadence).
- Each chunk's canvases are fed through the LLaVA-OV-2 processor's **image** path (`{'type': 'image', 'image': PIL.Image}`), because the OneVision encoder is purely spatial and the processor's video-backend wrapper itself goes through the image path anyway. Per-chunk `<Time=a-b s>` text tag is reused; we do not use the codec's per-patch `<X.X seconds>` tag rewriting in the chunk-streaming workflow.
- `train.py` is untouched (it already handles `model_type == 'llava_onevision2'`). The only changes are: a new helper module for codec-cache slicing, a new `video_backend` arg threaded through `DataArguments` → `LMMDataset`, and the `preprocess_conversation_stream` codec branch.
- An optional one-shot precompute script lets users prewarm the cache before training; it just calls the same `process_codec_video` in a for-loop.

**Tech Stack:** transformers 5.7.0 (already pinned), torch 2.7.1, flash-attn 2.8.0.post2, DeepSpeed 0.17.1, the bundled `codec_video_processing_llava_onevision2.py` from the LLaVA-OV-2 checkpoint, plus two new pip deps for codec preprocessing: `codec-video-prep` (provides the `cv-preinfer` CLI) and `opencv-python`, and a working `ffmpeg` 4.4.x–7.x on `$PATH`.

---

## Spec (settled before planning)

### S1. Why a separate backend, not a flag toggle

The frames backend pre-decodes the full clip into `Tensor[T,3,H,W]` and slices it uniformly by `streaming_fps_frames`. The codec backend produces:
- A list of N PIL canvases (each canvas = `images_per_group=4` raw frames mosaic'd into one RGB image, default `target_canvas=32` canvases → 32 PIL images per video).
- `src_positions: ndarray[N*ppc, 3]` where each row is `(t, h_patch, w_patch)` for one 14×14 patch on a canvas; `t` is the source frame index in the original video.
- `fps`, used to convert `t` (frame index) → seconds.

These two output shapes are incompatible inside one preprocessing function without conditionals. The cleanest design is to keep `preprocess_conversation_stream` and add a `_preprocess_conversation_stream_codec` sibling, picked by `self.video_backend`. The shared logic (text_stream → phrase, qa_stream, building the multi-round user/assistant list) is factored into helpers.

### S2. Time-windowing canvases

Each chunk in the current pipeline spans `streaming_fps_frames / FPS` seconds (default 2 frames / 2 FPS = 1 s).

Each codec canvas has a time range derived from its `ppc` patch rows in `src_positions`:
- `canvas_t_min_sec = src_positions[i*ppc:(i+1)*ppc, 0].min() / fps`
- `canvas_t_max_sec = src_positions[i*ppc:(i+1)*ppc, 0].max() / fps`

Each canvas covers approximately `group_size / fps ≈ 1 s` of source video by default (default `group_size=32`, source `fps=30`). That happens to align well with a 1-second chunk.

**Assignment rule (simple, deterministic):**
For chunk window `[chunk_start_sec, chunk_end_sec)`, include canvas `i` if its **midpoint** `(canvas_t_min_sec + canvas_t_max_sec) / 2` falls in `[chunk_start_sec, chunk_end_sec)`. This guarantees every canvas is assigned to exactly one chunk (no duplicates, no gaps) and is robust to small timestamp drift between codec sampling rate and chunk size.

Chunks with zero assigned canvases are dropped from the dialog (no round emitted for them), the same as how the frames backend handles boundaries.

### S3. Per-chunk processor call

Frames-backend chunk emits:
```python
user_content = [
    {'type': 'text',  'text': f'Time={a:.1f}-{b:.1f}s {optional_question}'},
    {'type': 'video', 'video': tensor_T3HW_chunk},
]
```

Codec-backend chunk emits:
```python
user_content = [
    {'type': 'text',  'text': f'Time={a:.1f}-{b:.1f}s {optional_question}'},
    {'type': 'image', 'image': canvas_pil_0},
    {'type': 'image', 'image': canvas_pil_1},
    ...
]
```

`LlavaOnevision2Processor.__call__(text=..., images=[PIL, ...], return_tensors='pt')` handles multi-image input natively (its image-path branch at `processing_llava_onevision2.py:421-472` expands `<|image_pad|>` per image using `image_grid_thw`). Each `<|image_pad|>` in the chat template that the jinja produces for a `{'type': 'image'}` content item gets expanded into the right number of merged tokens for that canvas. No need to call `rewrite_text_with_codec_positions` — that function is only useful when you want per-patch source timestamps embedded in the text, which is the codec's *inference* mode, not training where we already have per-chunk time tags.

### S4. Where `cv-preinfer` runs

On-the-fly inside the dataloader worker, via the existing `process_codec_video(video_url, cfg)` entry point. Its `flock`-protected on-disk cache (default `$HF_HOME/online_codec/`) handles concurrency: only the first worker to see a fresh video runs the actual `cv-preinfer` subprocess; everyone else blocks on the flock and then reads the cache.

If a user wants to pre-warm the cache (cheaper aggregate CPU usage, no first-epoch slowdown), they can run an optional `scripts/precompute_codec_cache.py` script that just iterates jsonl rows and calls `process_codec_video`. The training code does not care whether the cache was pre-warmed or filled live.

### S5. Configuration surface (added to `DataArguments`)

```python
@dataclass
class DataArguments:
    ...
    video_backend: str = 'frames'              # 'frames' (existing) | 'codec' (new)
    codec_target_canvas: int = 32              # cv-preinfer target_canvas
    codec_group_size: int = 32
    codec_images_per_group: int = 4
    codec_max_pixels: int = 150000             # per-canvas pixel budget
    codec_cache_root: str = ''                 # '' = use $ONLINE_CODEC_CACHE_DIR / $HF_HOME default
```

These are forwarded into a `CodecConfig` (defined inside the bundled `codec_video_processing_llava_onevision2`) and into our slicing helper.

The codec backend ignores `VIDEO_MIN_PIXELS / VIDEO_MAX_PIXELS / FPS_MAX_FRAMES` env vars (those control decord + smart_resize in the frames backend). The codec pixel budget lives entirely in `codec_max_pixels`.

### S6. Non-goals

- Inference adaptation: not in this plan. The frames backend StreamingVLM-style inference patches in `streaming_vlm/inference/qwen2_5/*` already do not apply to LLaVA-OV-2; we'd need a parallel `streaming_vlm/inference/llava_onevision2/` track for either backend, but that's separate work.
- Stage 2 (high-quality fg annealing) for codec backend: trivially obtained by copying the Stage 1 launch script and swapping the jsonl list + model path. Not detailed here.
- DDP-aware codec cache prefilling: the optional precompute script is single-process. Users wanting multi-GPU prewarming can run multiple copies pointing at disjoint jsonl shards; this is so simple it doesn't need framework support.
- Inf-Stream-Eval / OVO / VQA / LiveSports3k: those eval pipelines use the inference path, out of scope.

### S7. Acceptance criteria

1. `pip install codec-video-prep opencv-python` in `.venv-sft` succeeds and `cv-preinfer --help` returns 0.
2. `python -c "from codec_video_processing_llava_onevision2 import process_codec_video, CodecConfig; ..."` (after copying the bundled file under `streaming_vlm/` or adding the model dir to `sys.path`) returns a result dict with `images`, `src_positions`, `fps`.
3. New `streaming_vlm/data/codec_slicing.py` exposes `slice_canvases_by_time(images, src_positions, fps, chunk_starts_sec, chunk_ends_sec, ppc) -> list[list[PIL.Image]]` and is unit-tested.
4. `LMMDataset.preprocess_conversation_stream` (codec branch) produces a `conversation` shaped identically to the frames branch (same role order, same number of rounds = number of non-empty chunks, same `text_stream` slicing).
5. `LMMDataset.getitem(0)` on a real Inf-Stream-Train sample with `video_backend='codec'` returns a `BatchFeature` with `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `patch_positions`, `labels`; `labels` has more than zero non-`-100` entries.
6. `model(**inputs)` returns a finite loss tensor.
7. `torchrun --standalone --nproc_per_node=4 train.py --video_backend codec ... --max_steps 2` produces `checkpoint-2/trainer_state.json` with two finite, positive loss values, on the 4× RTX A6000 dev box.
8. Optional: `scripts/precompute_codec_cache.py path/to/train.jsonl` runs to completion and the resulting cache hits during training (worker-side `process_codec_video` returns immediately without running `cv-preinfer`).

### S8. What the existing frames-backend plan already provides

This plan **assumes** the prior `docs/superpowers/plans/2026-05-27-llava-ov2-sft.md` is fully landed (Tasks 0-6 there are already complete on `main`). Specifically:
- `train.py` already branches on `model_type` and loads LO2 cleanly.
- `LMMDataset.__init__` already recognizes `LlavaOnevision2Processor` and sets `model_base='LlavaOnevision2'`.
- `_video_tensor_to_np_frames` helper exists.
- The `save_pretrained` shim, `past_index` getattr fix, and `--overwrite_output_dir` removal are committed.

If any of those is missing, run that plan first.

---

## File Structure

**Modify:**
- `streaming_vlm/data/lmm_dataset.py` — add `video_backend`, `codec_*` to `DataArguments`; add `_preprocess_conversation_stream_codec` method; branch in `getitem`; add small helpers to share logic with the frames branch. The file already has the LO2 detection branch and `_video_tensor_to_np_frames` from the prior plan. Expected growth: +~120 lines.
- `train.py` — no changes. (`DataArguments` is consumed via `asdict(data_args)` already, so the new fields flow through automatically.)
- `scripts/sft_stage_1_llavaov2.sh` — no changes; users override via CLI flags or env vars on the existing script.

**Create:**
- `streaming_vlm/data/codec_slicing.py` — pure-function module:
  - `compute_canvas_time_ranges(src_positions: np.ndarray, ppc: int, fps: float) -> np.ndarray[N, 2]` returns `(t_min_sec, t_max_sec)` per canvas.
  - `slice_canvases_by_time(images, src_positions, fps, chunk_starts_sec, chunk_ends_sec, ppc) -> list[list[PIL.Image]]` returns per-chunk canvas lists, midpoint-assigned.
- `streaming_vlm/data/codec_loader.py` — thin wrapper over the bundled `codec_video_processing_llava_onevision2`:
  - `load_codec_payload(video_url, codec_config: dict) -> dict` builds a `CodecConfig` and calls `process_codec_video`, returning `{'images', 'src_positions', 'fps', 'ppc'}` where `ppc = src_positions.shape[0] // len(images)` (after dropping padding canvases).
- `scripts/precompute_codec_cache.py` — optional, ~40 lines: iterate one jsonl, call `load_codec_payload` for each video, exit on first failure with a clear message.
- `scripts/sft_stage_1_llavaov2_codec.sh` — copy of the frames Stage-1 launch script with `--video_backend codec` and codec knobs added.
- `tests/data/test_codec_slicing.py` — pytest covering `compute_canvas_time_ranges` and `slice_canvases_by_time` with synthetic inputs (no real video needed).

Why this split:
- `codec_slicing.py` is pure math (numpy/PIL only). Easy to unit-test, no video deps.
- `codec_loader.py` isolates the dynamic-import dance of pulling code out of the model's `trust_remote_code` directory.
- Keeping the entry point as one new method on `LMMDataset` (`_preprocess_conversation_stream_codec`) parallels the existing `preprocess_conversation_stream` and makes the dispatch trivial in `getitem`.

---

## Pre-flight

### Task 0: Environment + cv-preinfer install

**Files:** none.

- [ ] **Step 1: Install codec deps into the existing venv**

```bash
VIRTUAL_ENV=/data/v-kaichen/streaming-vlm/.venv-sft \
  uv pip install codec-video-prep opencv-python
```
Expected: both packages install. `codec-video-prep` provides the `cv-preinfer` CLI.

- [ ] **Step 2: Confirm cv-preinfer + ffmpeg are reachable**

```bash
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import shutil, subprocess
assert shutil.which('cv-preinfer'), 'cv-preinfer not on PATH'
assert shutil.which('ffmpeg'), 'ffmpeg not on PATH'
print('cv-preinfer:', subprocess.run(['cv-preinfer', '--help'], capture_output=True, text=True, timeout=10).returncode)
print('ffmpeg:', subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10).stdout.split('\\n')[0])
"
```
Expected: `cv-preinfer: 0`, `ffmpeg: ffmpeg version 4.x` (or 5/6/7).x.

- [ ] **Step 3: If ffmpeg is missing**

The dev box may not have ffmpeg system-installed. Install with:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```
Re-run Step 2. If that fails too (no sudo, no apt), report BLOCKED.

- [ ] **Step 4: One-shot validation that `process_codec_video` works on a real video**

```bash
DATASET_PATH=/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \
PYTHONPATH=/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
from codec_video_processing_llava_onevision2 import CodecConfig, process_codec_video
from pathlib import Path
import os, tempfile

cfg = CodecConfig(cache_root=Path(tempfile.mkdtemp(prefix='codec_smoke_')))
url = '/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train/Livecc_sft/-7qNuHVHqcU_8.25-173.02_2.0fps.mp4'
assert os.path.isfile(url), url
out = process_codec_video(url, cfg)
print('canvases:', len(out['images']))
print('src_positions shape:', out['src_positions'].shape)
print('fps:', out['fps'])
assert len(out['images']) > 0 and out['src_positions'].shape[1] == 3
print('OK')
"
```
Expected: `canvases: <some N up to 32>`, `src_positions shape: (<N*ppc>, 3)`, `fps:` a positive float, `OK`. May take 10-60 seconds the first time (cv-preinfer decodes the video).

No commit for this task. Pure environment validation.

---

## Code changes

### Task 1: Pure-function canvas slicing module

**Files:**
- Create: `streaming_vlm/data/codec_slicing.py`
- Create: `tests/data/test_codec_slicing.py`

- [ ] **Step 1: Write the failing tests**

`tests/data/test_codec_slicing.py`:

```python
import numpy as np
from PIL import Image
import pytest

from streaming_vlm.data.codec_slicing import (
    compute_canvas_time_ranges,
    slice_canvases_by_time,
)


def _synthetic_positions(n_canvases: int, ppc: int, first_frame_per_canvas: list[int]):
    """src_positions for N canvases, each with ppc patches spread across
    ``images_per_group=4`` consecutive source frames starting at first_frame.
    Mimics what cv-preinfer emits.
    """
    assert len(first_frame_per_canvas) == n_canvases
    assert ppc % 4 == 0, "test helper assumes images_per_group=4 so ppc%4==0"
    rows_per_subframe = ppc // 4
    out = []
    for first in first_frame_per_canvas:
        for sub in range(4):
            t = first + sub
            for k in range(rows_per_subframe):
                out.append((t, k % 16, k // 16))
    return np.array(out, dtype=np.int64)


def test_compute_canvas_time_ranges_basic():
    src_positions = _synthetic_positions(
        n_canvases=3, ppc=16, first_frame_per_canvas=[0, 30, 60],
    )
    fps = 30.0
    ranges = compute_canvas_time_ranges(src_positions, ppc=16, fps=fps)
    assert ranges.shape == (3, 2)
    # canvas 0: frames 0..3 → 0.0..0.1 sec
    np.testing.assert_allclose(ranges[0], [0.0, 3.0 / 30.0])
    np.testing.assert_allclose(ranges[1], [30.0 / 30.0, 33.0 / 30.0])
    np.testing.assert_allclose(ranges[2], [60.0 / 30.0, 63.0 / 30.0])


def test_slice_canvases_by_time_midpoint_assignment():
    src_positions = _synthetic_positions(
        n_canvases=4, ppc=16, first_frame_per_canvas=[0, 30, 60, 90],
    )
    # Each canvas mid: 0.05 / 1.05 / 2.05 / 3.05 sec
    images = [Image.new("RGB", (28, 28), (i, i, i)) for i in range(4)]
    chunks = slice_canvases_by_time(
        images=images,
        src_positions=src_positions,
        fps=30.0,
        chunk_starts_sec=[0.0, 1.0, 2.0, 3.0],
        chunk_ends_sec=[1.0, 2.0, 3.0, 4.0],
        ppc=16,
    )
    assert len(chunks) == 4
    assert [len(c) for c in chunks] == [1, 1, 1, 1]
    # First canvas (color 0) should be in chunk 0
    assert chunks[0][0].getpixel((0, 0)) == (0, 0, 0)
    assert chunks[3][0].getpixel((0, 0)) == (3, 3, 3)


def test_slice_canvases_by_time_empty_chunk():
    """A chunk window with no canvas midpoints inside gets an empty list."""
    src_positions = _synthetic_positions(
        n_canvases=2, ppc=16, first_frame_per_canvas=[0, 90],
    )
    images = [Image.new("RGB", (28, 28), c) for c in [(1, 1, 1), (2, 2, 2)]]
    chunks = slice_canvases_by_time(
        images=images, src_positions=src_positions, fps=30.0,
        chunk_starts_sec=[0.0, 1.0, 2.0],
        chunk_ends_sec=[1.0, 2.0, 3.0],
        ppc=16,
    )
    assert [len(c) for c in chunks] == [1, 0, 0]
    # Canvas 1 (midpoint 3.05 sec) is outside all three chunk windows


def test_slice_canvases_by_time_multiple_per_chunk():
    src_positions = _synthetic_positions(
        n_canvases=3, ppc=16, first_frame_per_canvas=[0, 15, 30],
    )
    # Mids: 0.05, 0.55, 1.05 — first two go in chunk 0, third in chunk 1
    images = [Image.new("RGB", (28, 28), c) for c in [(1, 1, 1), (2, 2, 2), (3, 3, 3)]]
    chunks = slice_canvases_by_time(
        images=images, src_positions=src_positions, fps=30.0,
        chunk_starts_sec=[0.0, 1.0], chunk_ends_sec=[1.0, 2.0], ppc=16,
    )
    assert [len(c) for c in chunks] == [2, 1]


def test_slice_canvases_validates_ppc():
    src_positions = np.zeros((10, 3), dtype=np.int64)
    images = [Image.new("RGB", (28, 28))]
    with pytest.raises(ValueError, match="ppc"):
        slice_canvases_by_time(
            images=images, src_positions=src_positions, fps=30.0,
            chunk_starts_sec=[0.0], chunk_ends_sec=[1.0], ppc=16,
        )
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd /data/v-kaichen/streaming-vlm
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -m pytest tests/data/test_codec_slicing.py -v
```
Expected: `ImportError: No module named 'streaming_vlm.data.codec_slicing'` (or a pytest collection failure).

- [ ] **Step 3: Implement the module**

`streaming_vlm/data/codec_slicing.py`:

```python
"""Pure helpers for slicing codec-emitted canvases by time window.

These functions take the output of `process_codec_video` (after dropping
padding canvases) and group the canvases into chunks defined by
``[chunk_starts_sec[i], chunk_ends_sec[i])`` windows.

A canvas is assigned to the chunk whose window contains its midpoint
``(t_min + t_max) / 2``. This is deterministic, prevents canvas duplication
across chunks, and is robust to small timestamp drift.

No torch dependency; numpy + PIL only.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PIL import Image


def compute_canvas_time_ranges(
    src_positions: np.ndarray, ppc: int, fps: float,
) -> np.ndarray:
    """Return per-canvas ``(t_min_sec, t_max_sec)`` from per-patch positions.

    ``src_positions`` is the ``(N*ppc, 3)`` int64 array of ``(t_frame, h, w)``
    rows emitted by ``process_codec_video``. Time is the source-frame index.

    Returns ``np.ndarray[N, 2]`` of float seconds.
    """
    if src_positions.ndim != 2 or src_positions.shape[1] != 3:
        raise ValueError(
            f"src_positions must be (N*ppc, 3); got shape {src_positions.shape}"
        )
    total = src_positions.shape[0]
    if ppc <= 0 or total % ppc != 0:
        raise ValueError(
            f"src_positions length {total} not divisible by ppc={ppc}"
        )
    n = total // ppc
    t_frames = src_positions[:, 0].reshape(n, ppc).astype(np.float64)
    t_min_sec = t_frames.min(axis=1) / float(fps)
    t_max_sec = t_frames.max(axis=1) / float(fps)
    return np.stack([t_min_sec, t_max_sec], axis=1)


def slice_canvases_by_time(
    images: Sequence[Image.Image],
    src_positions: np.ndarray,
    fps: float,
    chunk_starts_sec: Sequence[float],
    chunk_ends_sec: Sequence[float],
    ppc: int,
) -> List[List[Image.Image]]:
    """Group canvases into per-chunk lists by midpoint time assignment.

    Returns ``[chunk_0_canvases, chunk_1_canvases, ...]`` of equal length
    to ``chunk_starts_sec``. Chunks with no assigned canvas get ``[]``.
    """
    if len(chunk_starts_sec) != len(chunk_ends_sec):
        raise ValueError("chunk_starts_sec and chunk_ends_sec length mismatch")
    n_canvases = len(images)
    if n_canvases == 0:
        return [[] for _ in chunk_starts_sec]
    ranges = compute_canvas_time_ranges(src_positions, ppc=ppc, fps=fps)
    if ranges.shape[0] != n_canvases:
        raise ValueError(
            f"derived canvas count {ranges.shape[0]} != len(images) {n_canvases} "
            f"(src_positions length {src_positions.shape[0]} / ppc {ppc})"
        )
    midpoints = ranges.mean(axis=1)
    out: List[List[Image.Image]] = [[] for _ in chunk_starts_sec]
    for canvas_idx, mid in enumerate(midpoints):
        for chunk_idx, (a, b) in enumerate(zip(chunk_starts_sec, chunk_ends_sec)):
            if a <= mid < b:
                out[chunk_idx].append(images[canvas_idx])
                break
    return out
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd /data/v-kaichen/streaming-vlm
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -m pytest tests/data/test_codec_slicing.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/codec_slicing.py tests/data/test_codec_slicing.py
git commit -m "feat(data): add pure-fn codec canvas time-window slicing helpers"
```

---

### Task 2: Codec payload loader

**Files:**
- Create: `streaming_vlm/data/codec_loader.py`

The model's bundled `codec_video_processing_llava_onevision2.py` lives in a `trust_remote_code` directory, not on the normal `sys.path`. We dynamically import it.

- [ ] **Step 1: Implement the loader**

`streaming_vlm/data/codec_loader.py`:

```python
"""Thin wrapper around the model checkpoint's bundled codec preprocessing.

Loads ``codec_video_processing_llava_onevision2.py`` from the LLaVA-OV-2
checkpoint directory and exposes a single function::

    load_codec_payload(video_url, codec_config) -> {
        'images': list[PIL.Image],     # padding-dropped
        'src_positions': np.ndarray,   # (len(images) * ppc, 3) int64
        'fps': float,
        'ppc': int,                    # patches per canvas
    }

The cache root is read from ``codec_config.get('cache_root')`` if set,
else from the upstream defaults (``$ONLINE_CODEC_CACHE_DIR`` /
``$HF_HOME/online_codec``). Cache hits return in tens of milliseconds;
cache misses run ``cv-preinfer`` (single-flight via flock).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

_BUNDLED_MODULE_NAME = "codec_video_processing_llava_onevision2"
_codec_mod = None


def _import_codec_module(checkpoint_dir: str):
    global _codec_mod
    if _codec_mod is not None:
        return _codec_mod
    src_path = Path(checkpoint_dir) / f"{_BUNDLED_MODULE_NAME}.py"
    if not src_path.is_file():
        raise FileNotFoundError(
            f"Codec module not found at {src_path}. Pass a valid LLaVA-OV-2 "
            "checkpoint directory as `codec_checkpoint_dir` in codec_config."
        )
    # Also make sibling files importable in case the module does relative-style imports.
    sib = str(src_path.parent)
    if sib not in sys.path:
        sys.path.insert(0, sib)
    spec = importlib.util.spec_from_file_location(_BUNDLED_MODULE_NAME, src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _codec_mod = mod
    return mod


def load_codec_payload(
    video_url: str,
    codec_config: Mapping[str, Any],
    checkpoint_dir: Optional[str] = None,
) -> dict:
    """Run codec preprocessing (cache-aware) and return canvases + positions.

    Parameters
    ----------
    video_url
        Absolute path to the source video file.
    codec_config
        Maps to ``CodecConfig`` fields. Recognized keys:
        ``target_canvas``, ``group_size``, ``images_per_group``, ``patch``,
        ``max_pixels``, ``min_group_frames``, ``max_group_frames``,
        ``spatial_mask_mode``, ``cache_root`` (str path), ``timeout_seconds``.
        Unknown keys are passed through (dataclass will reject them with a
        TypeError if invalid).
    checkpoint_dir
        Path to the LLaVA-OV-2 checkpoint dir (where the bundled
        ``codec_video_processing_llava_onevision2.py`` lives). If None,
        read from ``codec_config['codec_checkpoint_dir']``.

    Returns a dict with keys ``images`` (list[PIL.Image], padding canvases
    already dropped), ``src_positions`` (numpy int64, length = len(images)*ppc),
    ``fps`` (float), ``ppc`` (int).
    """
    if checkpoint_dir is None:
        checkpoint_dir = codec_config.get("codec_checkpoint_dir")
    if not checkpoint_dir:
        raise ValueError(
            "load_codec_payload needs a checkpoint_dir (where the codec module lives). "
            "Pass it explicitly or set codec_config['codec_checkpoint_dir']."
        )
    mod = _import_codec_module(checkpoint_dir)

    # Build CodecConfig from the dict (filter to known fields).
    cfg_kwargs = {}
    cfg_fields = {f.name for f in mod.CodecConfig.__dataclass_fields__.values()}
    for k, v in codec_config.items():
        if k in cfg_fields:
            cfg_kwargs[k] = v
    if "cache_root" in cfg_kwargs and isinstance(cfg_kwargs["cache_root"], str):
        if cfg_kwargs["cache_root"]:
            cfg_kwargs["cache_root"] = Path(cfg_kwargs["cache_root"])
        else:
            cfg_kwargs.pop("cache_root")
    cfg = mod.CodecConfig(**cfg_kwargs)

    payload = mod.process_codec_video(video_url, cfg)
    images, src_positions, _dropped = mod.drop_padding_canvases(
        payload["images"], payload["src_positions"],
    )
    if not images:
        raise RuntimeError(f"codec produced no usable canvases for {video_url}")
    total = src_positions.shape[0]
    if total % len(images) != 0:
        raise RuntimeError(
            f"codec patch-position rows {total} not divisible by canvas count {len(images)}"
        )
    ppc = total // len(images)
    return {
        "images": images,
        "src_positions": np.asarray(src_positions, dtype=np.int64),
        "fps": float(payload["fps"]),
        "ppc": ppc,
    }
```

- [ ] **Step 2: Smoke-test against the same video Task 0 used**

```bash
DATASET_PATH=/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import sys; sys.path.insert(0, '/data/v-kaichen/streaming-vlm')
from streaming_vlm.data.codec_loader import load_codec_payload

ck = '/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct'
url = '/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train/Livecc_sft/-7qNuHVHqcU_8.25-173.02_2.0fps.mp4'
out = load_codec_payload(url, codec_config={'cache_root': '/tmp/codec_smoke'}, checkpoint_dir=ck)
print('canvases:', len(out['images']))
print('src_positions:', out['src_positions'].shape, out['src_positions'].dtype)
print('fps:', out['fps'])
print('ppc:', out['ppc'])
assert len(out['images']) * out['ppc'] == out['src_positions'].shape[0]
print('OK')
"
```
Expected: `canvases: <N>`, `src_positions: (N*ppc, 3) int64`, `fps: <float>`, `ppc: <int>`, `OK`. First call slow (cv-preinfer runs), second call fast (cache hit).

- [ ] **Step 3: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/codec_loader.py
git commit -m "feat(data): add codec_loader wrapping bundled cv-preinfer pipeline"
```

---

### Task 3: Extend `DataArguments` with codec knobs

**Files:**
- Modify: `streaming_vlm/data/lmm_dataset.py:29-37` (the `DataArguments` dataclass).

- [ ] **Step 1: Read the current dataclass**

```bash
sed -n '29,40p' /data/v-kaichen/streaming-vlm/streaming_vlm/data/lmm_dataset.py
```

Confirm it currently contains:
```python
@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    text_sink:int = 0
    text_sliding_window:int = 0
```

- [ ] **Step 2: Replace it with the extended version**

```python
@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    text_sink: int = 0
    text_sliding_window: int = 0
    # ---- Video backend selection ----
    video_backend: str = 'frames'              # 'frames' (default) | 'codec'
    # ---- Codec backend knobs (ignored when video_backend='frames') ----
    codec_target_canvas: int = 32
    codec_group_size: int = 32
    codec_images_per_group: int = 4
    codec_max_pixels: int = 150000
    codec_cache_root: str = ''                 # '' = use $ONLINE_CODEC_CACHE_DIR / $HF_HOME default
    codec_checkpoint_dir: str = ''             # path to LLaVA-OV-2 checkpoint dir (where codec module lives)
```

- [ ] **Step 3: Smoke-import**

```bash
cd /data/v-kaichen/streaming-vlm
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
from streaming_vlm.data.lmm_dataset import DataArguments
import dataclasses
d = DataArguments()
print('video_backend:', d.video_backend, 'codec_target_canvas:', d.codec_target_canvas)
assert d.video_backend == 'frames'
print('OK')
"
```
Expected: `video_backend: frames codec_target_canvas: 32`, `OK`.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/lmm_dataset.py
git commit -m "feat(dataset): expose video_backend + codec knobs on DataArguments"
```

---

### Task 4: Thread the codec args into `LMMDataset.__init__`

**Files:**
- Modify: `streaming_vlm/data/lmm_dataset.py` — `LMMDataset.__init__`.

- [ ] **Step 1: Read the current __init__ signature and constants storage**

```bash
sed -n '73,130p' /data/v-kaichen/streaming-vlm/streaming_vlm/data/lmm_dataset.py
```

Confirm the constructor currently stores `processor`, `with_context`, `initial_fps_frames`, `streaming_fps_frames`, `text_sink`, `text_sliding_window` as attrs.

- [ ] **Step 2: Add codec args to the `__init__` signature**

In `LMMDataset.__init__`, between the existing kwargs and the `**kwargs` catch, add:

```python
        video_backend: str = DataArguments.video_backend,
        codec_target_canvas: int = DataArguments.codec_target_canvas,
        codec_group_size: int = DataArguments.codec_group_size,
        codec_images_per_group: int = DataArguments.codec_images_per_group,
        codec_max_pixels: int = DataArguments.codec_max_pixels,
        codec_cache_root: str = DataArguments.codec_cache_root,
        codec_checkpoint_dir: str = DataArguments.codec_checkpoint_dir,
```

The original signature ended with `**kwargs` so extra fields are accepted; adding them as explicit args lets us reference them as attrs.

- [ ] **Step 3: Store them as attributes**

At the bottom of `__init__` (after the existing `self.text_sliding_window = text_sliding_window` line), add:

```python
        self.video_backend = video_backend
        self.codec_config = {
            'target_canvas': codec_target_canvas,
            'group_size': codec_group_size,
            'images_per_group': codec_images_per_group,
            'max_pixels': codec_max_pixels,
        }
        if codec_cache_root:
            self.codec_config['cache_root'] = codec_cache_root
        self.codec_checkpoint_dir = codec_checkpoint_dir
        if self.video_backend == 'codec' and not self.codec_checkpoint_dir:
            raise ValueError(
                "video_backend='codec' requires codec_checkpoint_dir to be set "
                "(path to the LLaVA-OV-2 checkpoint dir containing "
                "codec_video_processing_llava_onevision2.py)."
            )
```

- [ ] **Step 4: Smoke-construct with both backends**

```bash
cd /data/v-kaichen/streaming-vlm
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import sys; sys.path.insert(0, '/data/v-kaichen/streaming-vlm')
from transformers import AutoProcessor
from streaming_vlm.data.lmm_dataset import LMMDataset

p = AutoProcessor.from_pretrained('/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct', trust_remote_code=True)
d_frames = LMMDataset(train_annotation_paths=[], processor=p, video_backend='frames')
assert d_frames.video_backend == 'frames'

d_codec = LMMDataset(
    train_annotation_paths=[], processor=p, video_backend='codec',
    codec_checkpoint_dir='/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct',
)
assert d_codec.video_backend == 'codec'
print('codec_config:', d_codec.codec_config)
print('OK')

try:
    LMMDataset(train_annotation_paths=[], processor=p, video_backend='codec')
except ValueError as e:
    print('correctly rejected missing checkpoint_dir:', e)
else:
    raise AssertionError('should have raised')
"
```
Expected: `codec_config: {...}`, `OK`, and the rejection check passes.

- [ ] **Step 5: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/lmm_dataset.py
git commit -m "feat(dataset): thread codec args into LMMDataset.__init__"
```

---

### Task 5: Implement `_preprocess_conversation_stream_codec`

**Files:**
- Modify: `streaming_vlm/data/lmm_dataset.py` — add a new method.

This method mirrors `preprocess_conversation_stream` step-by-step, but pulls frames from the codec payload instead of decord, and emits per-chunk `image` content items instead of one `video` tensor.

- [ ] **Step 1: Add the helper imports**

At the top of `streaming_vlm/data/lmm_dataset.py`, after existing imports, add:

```python
from streaming_vlm.data.codec_loader import load_codec_payload
from streaming_vlm.data.codec_slicing import slice_canvases_by_time
```

- [ ] **Step 2: Add the new method on `LMMDataset`**

Add this method to the `LMMDataset` class, immediately after `preprocess_conversation_stream`:

```python
    def preprocess_conversation_stream_codec(self, conversation: list):
        """Codec-backend analogue of preprocess_conversation_stream.

        Produces the same multi-round conversation structure, but each chunk's
        user.content carries codec canvases (PIL images, multi-image style)
        instead of a sliced video tensor.
        """
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']

        user_video_dict, _user_query_dict = user_content
        video_start = user_video_dict['video_start']
        video_end = user_video_dict['video_end']

        assert 'video' in user_video_dict, (
            'Codec backend: first user content must contain video path information'
        )
        video_path = user_video_dict['video']
        # Resolve relative to DATASET_PATH (matches the frames-backend behavior).
        if not os.path.isabs(video_path):
            video_path = os.path.join(os.environ['DATASET_PATH'], video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"codec: video not found at {video_path}")

        assistant_text_stream = assistant_message['content'][0]['text_stream']
        qa_stream = assistant_message['content'][0]['qa_stream'] if 'qa_stream' in assistant_message['content'][0] else []

        # ----- Codec preprocessing (cache-aware) -----
        payload = load_codec_payload(
            video_path,
            codec_config=self.codec_config,
            checkpoint_dir=self.codec_checkpoint_dir,
        )
        canvases = payload['images']
        src_positions = payload['src_positions']
        fps = payload['fps']
        ppc = payload['ppc']

        # ----- Build chunk windows (mirrors frames-backend timing) -----
        chunk_starts: list[float] = []
        chunk_ends: list[float] = []
        chunk_starts.append(video_start)
        chunk_ends.append(video_start + self.initial_fps_frames / FPS)
        t = self.initial_fps_frames
        # Use video_end as upper bound; if codec returned canvases past video_end
        # they're dropped by the time-window slicing.
        while video_start + t / FPS < video_end:
            chunk_starts.append(video_start + t / FPS)
            chunk_ends.append(video_start + (t + self.streaming_fps_frames) / FPS)
            t += self.streaming_fps_frames

        # ----- Slice canvases into chunks -----
        # Canvas timestamps from codec are relative to source video t=0,
        # while the conversation's chunk windows are in (video_start, video_end)
        # which is the same coordinate system (video_start=0 for Inf-Stream-Train).
        per_chunk_canvases = slice_canvases_by_time(
            images=canvases,
            src_positions=src_positions,
            fps=fps,
            chunk_starts_sec=chunk_starts,
            chunk_ends_sec=chunk_ends,
            ppc=ppc,
        )

        # ----- Build the multi-round conversation -----
        conversation_out: list[dict] = []
        next_start_from = 0
        for chunk_idx, (a, b, chunk_canvases) in enumerate(
            zip(chunk_starts, chunk_ends, per_chunk_canvases)
        ):
            if not chunk_canvases:
                # Skip empty chunks entirely — no visual signal to ground on.
                # text_stream advances over this gap in the next non-empty chunk.
                continue
            phrase, next_start_from = get_phrase_before_timestamp(
                assistant_text_stream, b, start_from=next_start_from,
            )
            if qa_stream and a < qa_stream[0][1] and b >= qa_stream[0][1]:
                question = qa_stream[0][2]
                answer = qa_stream[0][3]
                qa_stream = qa_stream[1:]
            else:
                question = ''
                answer = ''

            user_content_round = [
                {'type': 'text', 'text': f'Time={a:.1f}-{b:.1f}s {question}'.rstrip()},
            ]
            for img in chunk_canvases:
                user_content_round.append({'type': 'image', 'image': img})
            assistant_content_round = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]

            conversation_out.extend([
                {'role': 'user', 'content': user_content_round},
                {'role': 'assistant', 'content': assistant_content_round},
            ])

        if not conversation_out:
            raise RuntimeError(
                f"codec: no non-empty chunks produced for {video_path}; "
                f"check codec_target_canvas/group_size vs video duration"
            )

        # No "image_inputs" return — they're embedded directly in conversation_out
        # as {'type':'image', 'image': PIL}. getitem() will pick them up by
        # walking conversation_out elements (preprocess_image returns identity
        # if element['image'] is already a PIL Image).
        return conversation_out
```

- [ ] **Step 3: Branch `getitem` to call the codec method**

Locate the block in `getitem` (around `lmm_dataset.py:275-281` per the current file) that reads:

```python
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)
```

Replace with:

```python
        if special_process_for_stream:
            if self.video_backend == 'codec':
                conversation = self.preprocess_conversation_stream_codec(conversation)
                video_inputs = None
                image_inputs = None  # Images are embedded in conversation directly.
            else:
                conversation, video_inputs = self.preprocess_conversation_stream(conversation)
                image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)
```

- [ ] **Step 4: Make the processor call codec-aware**

In `getitem`, locate the existing processor-call branch added in the prior plan (search for `self.model_base == 'LlavaOnevision2' and video_inputs is not None`).

When `self.video_backend == 'codec'`, `video_inputs` is `None`, but the conversation already carries `{'type': 'image', 'image': PIL}` items, which the processor's `apply_chat_template` + `__call__` handle natively. We need the existing dispatch to fall into the `else` branch (the default processor call without the LO2-specific tensor conversion), because `image_inputs=None / videos=None` is the right invocation for a multi-image-via-template scenario.

Verify by reading the dispatch:
```bash
sed -n '290,330p' /data/v-kaichen/streaming-vlm/streaming_vlm/data/lmm_dataset.py
```
The current branch condition `if self.model_base == 'LlavaOnevision2' and video_inputs is not None:` already correctly falls through to the `else` branch when `video_inputs is None`. **No code change here**, but verify by reading and confirm the branch logic is right.

- [ ] **Step 5: Walk conversation_out to extract images for processor call**

The processor's `apply_chat_template` produces text with `<|image_pad|>` placeholders for every `{'type': 'image'}` content. The processor's `__call__(images=...)` then expects a corresponding list of images.

Look at the existing else branch:
```python
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
```

When `image_inputs is None` and the conversation has embedded `{'type': 'image', 'image': PIL}` items, the processor will NOT auto-discover them. We must collect them.

Modify the codec branch in `getitem` (the one we just added in Step 3) to also collect the images:

Replace
```python
            if self.video_backend == 'codec':
                conversation = self.preprocess_conversation_stream_codec(conversation)
                video_inputs = None
                image_inputs = None  # Images are embedded in conversation directly.
```
with:
```python
            if self.video_backend == 'codec':
                conversation = self.preprocess_conversation_stream_codec(conversation)
                video_inputs = None
                # Extract PIL images from the embedded {'type':'image'} content items
                # in chat-template order. The processor will pair each <|image_pad|>
                # placeholder with the next image from this list.
                image_inputs = []
                for msg in conversation:
                    if msg.get('role') != 'user':
                        continue
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        for el in content:
                            if isinstance(el, dict) and el.get('type') == 'image':
                                image_inputs.append(el['image'])
                if not image_inputs:
                    image_inputs = None
```

- [ ] **Step 6: Smoke-test with a real sample**

```bash
DATASET_PATH=/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import sys; sys.path.insert(0, '/data/v-kaichen/streaming-vlm')
import os
from transformers import AutoProcessor
from streaming_vlm.data.lmm_dataset import LMMDataset

ck = '/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct'
p = AutoProcessor.from_pretrained(ck, trust_remote_code=True, padding_side='right')
d = LMMDataset(
    train_annotation_paths=[f'{os.environ[\"DATASET_PATH\"]}/train_livecc_with_seeks.jsonl'],
    processor=p, text_sink=512, text_sliding_window=512,
    video_backend='codec',
    codec_checkpoint_dir=ck,
    codec_cache_root='/tmp/codec_train_cache',
)
batch = d[0]
for k, v in batch.items():
    print(k, getattr(v, 'shape', type(v).__name__))
nz = (batch['labels'] != -100).sum().item()
print('labels non -100:', nz)
assert nz > 0, 'no supervised tokens'
print('OK')
"
```
Expected (first run can take 30+ seconds for codec):
- Keys printed: `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `patch_positions`, `labels`.
- `labels non -100` > 0.
- `OK`.

- [ ] **Step 7: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/lmm_dataset.py
git commit -m "feat(dataset): codec-backend chunked-streaming dialog construction"
```

---

### Task 6: Add codec launch script

**Files:**
- Create: `scripts/sft_stage_1_llavaov2_codec.sh`

- [ ] **Step 1: Copy and adapt the frames launch script**

```bash
cp /data/v-kaichen/streaming-vlm/scripts/sft_stage_1_llavaov2.sh \
   /data/v-kaichen/streaming-vlm/scripts/sft_stage_1_llavaov2_codec.sh
chmod +x /data/v-kaichen/streaming-vlm/scripts/sft_stage_1_llavaov2_codec.sh
```

- [ ] **Step 2: Edit the copy to add codec args**

Open `scripts/sft_stage_1_llavaov2_codec.sh`. Make these edits:

(a) Change the project name. Find:
```bash
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-StreamingVLM_LlavaOV2_SFT_stage_1}
```
Replace with:
```bash
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-StreamingVLM_LlavaOV2_codec_SFT_stage_1}
```

(b) Add a codec config block right after the existing video pixel/frame budget block (around the `text_sink=` lines). Insert:
```bash
# ---- Codec backend knobs ----
codec_target_canvas=${CODEC_TARGET_CANVAS:-32}
codec_group_size=${CODEC_GROUP_SIZE:-32}
codec_images_per_group=${CODEC_IMAGES_PER_GROUP:-4}
codec_max_pixels=${CODEC_MAX_PIXELS:-150000}
codec_cache_root=${CODEC_CACHE_ROOT:-}
codec_checkpoint_dir=${CODEC_CHECKPOINT_DIR:-/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct}
```

(c) Add the new CLI flags to the torchrun command. Find the line:
```bash
    --text_sliding_window $TEXT_SLIDING_WINDOW
```
Replace with:
```bash
    --text_sliding_window $TEXT_SLIDING_WINDOW \
    --video_backend codec \
    --codec_target_canvas $codec_target_canvas \
    --codec_group_size $codec_group_size \
    --codec_images_per_group $codec_images_per_group \
    --codec_max_pixels $codec_max_pixels \
    --codec_cache_root "$codec_cache_root" \
    --codec_checkpoint_dir "$codec_checkpoint_dir"
```

- [ ] **Step 3: Syntax check**

```bash
bash -n /data/v-kaichen/streaming-vlm/scripts/sft_stage_1_llavaov2_codec.sh && echo "syntax ok"
```
Expected: `syntax ok`.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add scripts/sft_stage_1_llavaov2_codec.sh
git commit -m "feat(scripts): add codec-backend Stage-1 SFT launch script"
```

---

### Task 7: Optional precompute script

**Files:**
- Create: `scripts/precompute_codec_cache.py`

- [ ] **Step 1: Write the script**

`scripts/precompute_codec_cache.py`:

```python
#!/usr/bin/env python
"""Pre-warm the codec on-disk cache for every video in a jsonl annotation file.

Run this before training to avoid first-epoch slowdowns. Single-process; for
multi-GPU prewarming, run multiple copies pointing at disjoint shards of the
jsonl (e.g., split with ``split -n l/4 file.jsonl shard_``).

Usage::

    python scripts/precompute_codec_cache.py \\
        --jsonl /path/to/train_livecc_with_seeks.jsonl \\
        --dataset-path /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \\
        --checkpoint-dir /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \\
        --cache-root /shared/codec_cache \\
        --max-pixels 150000

Idempotent: cache hits return immediately, so re-running is safe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streaming_vlm.data.codec_loader import load_codec_payload


def iter_video_urls(jsonl_path: str, dataset_path: str):
    seen = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, list):
                continue
            for msg in rec:
                if msg.get('role') == 'user':
                    for el in msg.get('content', []):
                        if isinstance(el, dict) and el.get('type') == 'video':
                            v = el.get('video')
                            if not v:
                                continue
                            if not os.path.isabs(v):
                                v = os.path.join(dataset_path, v)
                            if v not in seen and os.path.isfile(v):
                                seen.add(v)
                                yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--dataset-path', required=True)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--cache-root', default='')
    ap.add_argument('--target-canvas', type=int, default=32)
    ap.add_argument('--group-size', type=int, default=32)
    ap.add_argument('--images-per-group', type=int, default=4)
    ap.add_argument('--max-pixels', type=int, default=150000)
    args = ap.parse_args()

    cfg = {
        'target_canvas': args.target_canvas,
        'group_size': args.group_size,
        'images_per_group': args.images_per_group,
        'max_pixels': args.max_pixels,
    }
    if args.cache_root:
        cfg['cache_root'] = args.cache_root

    n_ok = 0
    n_skip = 0
    t0 = time.time()
    for url in iter_video_urls(args.jsonl, args.dataset_path):
        try:
            load_codec_payload(url, codec_config=cfg, checkpoint_dir=args.checkpoint_dir)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            print(f"[skip] {url}: {e}", file=sys.stderr)
        if (n_ok + n_skip) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[progress] {n_ok} ok / {n_skip} skipped, {elapsed:.1f}s")

    print(f"[done] {n_ok} ok / {n_skip} skipped in {time.time() - t0:.1f}s")
    return 1 if n_skip > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run it against a small subset to verify**

```bash
# Take first 3 records of livecc jsonl as a smoke shard
head -3 /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train/train_livecc_with_seeks.jsonl \
    > /tmp/codec_smoke_shard.jsonl

/data/v-kaichen/streaming-vlm/.venv-sft/bin/python \
    /data/v-kaichen/streaming-vlm/scripts/precompute_codec_cache.py \
    --jsonl /tmp/codec_smoke_shard.jsonl \
    --dataset-path /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \
    --checkpoint-dir /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
    --cache-root /tmp/codec_train_cache
```
Expected: at most 3 videos processed, `[done] 3 ok / 0 skipped` (or some skipped on missing files), exit 0.

Cleanup: `rm /tmp/codec_smoke_shard.jsonl`.

- [ ] **Step 3: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add scripts/precompute_codec_cache.py
git commit -m "feat(scripts): add optional codec cache prewarm script"
```

---

## Smoke test gate

### Task 8: 4-GPU 2-step smoke run with codec backend

**Files:** none new.

This is the integration gate. It exercises model load + codec dataset + forward + backward end-to-end.

- [ ] **Step 1: Prerequisites check**

```bash
test -d /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train && \
    test -f /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train/train_livecc_with_seeks.jsonl && \
    /data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "from streaming_vlm.data.codec_loader import load_codec_payload" && \
    /data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "from streaming_vlm.data.codec_slicing import slice_canvases_by_time" && \
    echo "ok"
```
Expected: `ok`.

- [ ] **Step 2: 4-GPU smoke**

```bash
cd /data/v-kaichen/streaming-vlm
rm -rf ./checkpoints/llavaov2_codec_smoke* 2>/dev/null || true

DATASET_PATH=/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
/data/v-kaichen/streaming-vlm/.venv-sft/bin/torchrun \
    --standalone --nproc_per_node=4 train.py \
    --deepspeed ./scripts/zero3.json \
    --output_dir ./checkpoints/llavaov2_codec_smoke \
    --run_name llavaov2_codec_smoke \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.0 \
    --optim adamw_torch \
    --lr_scheduler_type constant \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
    --train_annotation_paths /data/v-kaichen/streaming-vlm/data/Inf-Stream-Train/train_livecc_with_seeks.jsonl \
    --dataloader_num_workers 0 \
    --report_to none \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 1 \
    --prediction_loss_only true \
    --text_sink 512 \
    --text_sliding_window 512 \
    --video_backend codec \
    --codec_max_pixels 150000 \
    --codec_target_canvas 32 \
    --codec_group_size 32 \
    --codec_images_per_group 4 \
    --codec_cache_root /tmp/codec_train_cache \
    --codec_checkpoint_dir /data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct \
    2>&1 | tee /tmp/llavaov2_codec_smoke.log
```

Expected: process exits 0; 2 finite positive loss values logged; `./checkpoints/llavaov2_codec_smoke/checkpoint-2/trainer_state.json` exists.

- [ ] **Step 3: If it OOMs**

The codec backend's pixel budget is per canvas. Default 150000 px × ≤32 canvases per video × bf16 patches is comparable to a single moderately-large image. If OOM happens, lower in this order:
1. `--codec_max_pixels 100000`
2. `--codec_target_canvas 16` (halves canvas count)
3. `--codec_max_pixels 60000`

If still OOM after these, report BLOCKED with the trace.

- [ ] **Step 4: If you get "no non-empty chunks produced"**

This means every chunk window happens to fall between codec canvas midpoints. Two likely causes:
- The video is shorter than `initial_fps_frames / FPS` (= 1 s default). Try a different sample.
- The codec emitted very few canvases for a very short video. Try lowering `codec_target_canvas`.

If it persists across many samples, that's a real algorithm bug — report BLOCKED with the video URL and the codec payload sizes.

- [ ] **Step 5: Clean up**

```bash
rm -rf /data/v-kaichen/streaming-vlm/checkpoints/llavaov2_codec_smoke*
```

- [ ] **Step 6: Optionally commit working hyperparameters**

If you had to lower the budget to make the smoke work, edit `scripts/sft_stage_1_llavaov2_codec.sh` to reflect the tuned defaults and commit:

```bash
git add scripts/sft_stage_1_llavaov2_codec.sh
git commit -m "fix(scripts): tune codec budget defaults for 46 GB GPUs"
```

If no tuning was needed, no commit.

---

## Out of scope (explicitly)

- Stage 2 codec annealing: copy this Stage 1 script, swap jsonl + model path. Same plan applies.
- Multi-process / distributed precompute scheduler: out of scope; users can shard manually.
- Inference codec support: separate plan (the bundled processor already supports `video_backend="codec"` at inference, but the existing `streaming_vlm/inference/qwen2_5/*` patches don't apply to LO2 anyway, so this needs its own track).
- Mixed-backend training (some samples codec, some frames): out of scope. Pick one per training run.

---

## Self-review checklist

1. **Spec coverage:** S4 (config surface) → Task 3; S5 (where cv-preinfer runs) → Task 4 + Task 7; S2 + S3 (time-windowing + image emission) → Task 1 + Task 5; S7.1 → Task 0; S7.2 → Task 0; S7.3 → Task 1 tests; S7.4 + S7.5 → Task 5 Step 6; S7.6 → Task 8; S7.7 → Task 8; S7.8 → Task 7. All covered.
2. **Placeholder scan:** No "TBD", "implement later", "add validation as needed". Every code block is complete and runnable.
3. **Type/name consistency:**
   - `load_codec_payload(video_url, codec_config, checkpoint_dir=None)` signature used identically in Task 2, Task 5, Task 7.
   - `slice_canvases_by_time(images, src_positions, fps, chunk_starts_sec, chunk_ends_sec, ppc)` signature identical in Task 1 and Task 5.
   - `self.codec_config` (a dict in Task 4) and `self.codec_checkpoint_dir` (a str in Task 4) used in Task 5 unchanged.
   - `--video_backend codec` flag in Task 6 script and Task 8 smoke matches the `DataArguments.video_backend` field added in Task 3.
   - `'codec_cache_root'` as a config key consistently treated as a str path that becomes `Path` inside `load_codec_payload` (Task 2).
