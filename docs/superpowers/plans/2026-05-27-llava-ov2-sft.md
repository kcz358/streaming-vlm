# LLaVA-OneVision-2-8B-Instruct SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the StreamingVLM SFT pipeline to fine-tune `LLaVA-OneVision-2-8B-Instruct` (a Qwen3-8B language backbone + custom OneVision vision tower) on the existing Inf-Stream-Train corpus, using the same "multi-round streaming dialog + previous-text sink/sliding-window" data-level trick.

**Architecture:**
- Reuse `train.py` / `LMMDataset` / `compute_loss_logging_labels` / `get_qwen_range` / Trainer + DeepSpeed setup unchanged in spirit, but swap the model loading + processor + video tensor handling + label scanning to match LLaVA-OV-2's API.
- All model-side "patches" are bound directly onto the loaded instance via `MethodType`, not via class-level monkey patches (LLaVA-OV-2 has no shared `liger_kernel` hook and uses its own model_type `llava_onevision2`, so class-level patching is brittle).
- No streaming attention / RoPE / KV-evict code is touched: LLaVA-OV-2 uses 1D RoPE on Qwen3, has no `rope_deltas` / `get_rope_index`, so we delete those bindings entirely.

**Tech Stack:** transformers 5.7.0 (already upgraded in `.venv-sft`), flash-attn 2.8.0.post2, DeepSpeed 0.17.1, torch 2.7.1, the bundled `trust_remote_code` files inside the LLaVA-OV-2 checkpoint (`modeling_llava_onevision2.py`, `processing_llava_onevision2.py`, `video_processing_llava_onevision2.py`, `chat_template.jinja`).

**Data layout (already set up):**
- `./data/Inf-Stream-Train/` is a real directory containing symlinks: every entry of `/data/v-kaichen/azure_blob/data/Inf-Stream-Train/*` is linked in, plus an extra `Livecc_sft → /data/v-kaichen/azure_blob/data/Livecc_sft/video/youtube`. The Livecc jsonl entries reference `Livecc_sft/<id>.mp4` relative to `DATASET_PATH`, so that symlink resolves them.
- All training/eval scripts must therefore set `DATASET_PATH=$(pwd)/data/Inf-Stream-Train`.

---

## Spec (settled before planning)

### S1. Why simpler than Qwen2.5-VL

The Qwen2.5-VL SFT pipeline carries three model-level hacks: (1) replacing liger-kernel's `qwen2_5_vl.lce_forward` to fix `rope_deltas` reuse across epochs, (2) overriding `model.get_rope_index` for 3D mRoPE, and (3) `_update_causal_mask` / `streaming_text_flash_attn_forward` for inference. LLaVA-OV-2:
- Uses Qwen3 (`text_config.model_type = "qwen3"`) with **pure 1D RoPE**. `LlavaOnevision2Model.forward` (line 1185-1198) constructs `position_ids` as a plain `(B, L)` tensor and never asks for `rope_deltas` / mRoPE. → Drop (1) and (2) entirely.
- Computes loss itself via `self.loss_function(logits, labels, vocab_size, **kwargs)` (line 1383-1386). No liger patching needed for correctness; we just leave `--use_liger_kernel` off.
- We only train, never run streaming inference here. → Drop (3).

So the only model-level binding we need is the existing Trainer `compute_loss` patch for richer logging. Everything else is **data-shape adaptation**.

### S2. What must change vs the current Qwen2.5-VL SFT

| Concern | Qwen2.5-VL behavior | LLaVA-OV-2 behavior | Required change |
|---|---|---|---|
| Model class | `getattr(transformers, "Qwen2_5_VLForConditionalGeneration")` | Local `LlavaOnevision2ForConditionalGeneration`, registered via `auto_map` | Use `AutoModelForImageTextToText.from_pretrained(..., trust_remote_code=True)`. |
| Processor | `AutoProcessor` direct | Local `LlavaOnevision2Processor` (no `ProcessorMixin`) | Use `AutoProcessor.from_pretrained(..., trust_remote_code=True)`; do NOT fall back to Qwen2-VL processor. |
| Video input format | `videos=[torch.Tensor[T,3,H,W]]` accepted by qwen-vl-utils | `LlavaOnevision2VideoProcessor.__call__` accepts file path, `list[PIL.Image]`, or `list[np.ndarray]` (line 501-516). It does **not** accept `torch.Tensor`. | Convert per-chunk frame tensor `clip[i:i+streaming_fps_frames]` (uint8, `(T,3,H,W)`) to `list[np.ndarray]` shape `(H,W,3)` before passing through. |
| Video → image alias | Processor outputs `pixel_values_videos` / `video_grid_thw` consumed by model | Processor rewrites `<\|video_pad\|>` → per-frame `<X.X seconds><\|vision_start\|><\|image_pad\|>*n<\|vision_end\|>` blocks, and returns `pixel_values` / `image_grid_thw` / `patch_positions` (line 411-418). Model's `forward` ignores `pixel_values_videos`. | LMMDataset must propagate `patch_positions` through `data_collator`; otherwise no change. |
| Chat-template "previous text" role | jinja accepts any role string and emits `<\|im_start\|>{{role}}\n...<\|im_end\|>` | Same jinja shape — but the jinja unconditionally prepends a system message when `loop.first and message['role'] != 'system'`. With `previous text` as role 0, jinja will emit `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>previous text\n...<\|im_end\|>`. That is fine for `get_qwen_range('previous text', 0)` to still find the segment, but we need to verify the `im_start`/`im_end`/role-token IDs are the **same** Qwen tokenizer IDs that `get_qwen_range` hard-codes. | Verify with a tokenize sanity test; no expected code change. |
| Label-region detection | `get_qwen_range` finds `<\|im_start\|>assistant ... <\|im_end\|>`, plus assistant-mask via `assistant_id = "assistant"` token id, `im_start_id` / `im_end_id`. | Tokenizer is the same Qwen family (`added_tokens.json` shows identical `<\|im_start\|>=151644`, `<\|im_end\|>=151645`, `<\|vision_start\|>=151652`, `<\|vision_end\|>=151653`, `<\|image_pad\|>=151655`, `<\|video_pad\|>=151656`). | Verify; expected to work unchanged. |
| Tokenizer class detection | `LMMDataset.__init__` branches on `processor.__class__.__name__ in {Qwen2VL, Qwen2_5_VL}` | Class is `LlavaOnevision2Processor` | Add a third branch (or generalize) so model_base resolves correctly. |
| Vision tower freeze | `getattr(model, 'visual')` / `'vision_tower'` | `model.model.visual` (LlavaOnevision2Model holds the visual; LlavaOnevision2ForConditionalGeneration only proxies `.visual` as a property at line 1263-1265, but `requires_grad_(False)` still works through the property because it returns the actual module reference) | Add `getattr(model.model, 'visual', None)` fallback alongside existing names. |
| Liger fused linear CE | Class-level monkey patch | No corresponding hook | Remove the import + patch lines from `train.py` for this run; do not pass `--use_liger_kernel`. |
| `model.rope_deltas` | Exists, patched | Does not exist | Skip the binding. |
| `model.get_rope_index` | Bound | Does not exist | Skip the binding. |
| Embedding alias `llm_model_embed_tokens` | Present in Qwen2.5-VL nested layout | LLaVA-OV-2 exposes `model.language_model.embed_tokens` (Qwen3 layout) | Skip the `delattr` + property hack; it's specific to a prior model. |

### S3. Non-goals

- No streaming-mask / KV-evict / inference adaptation in this plan. The training run is enough to produce a Stage-1 ckpt; inference adaptation is a separate plan.
- No new evaluator. Reusing `compute_loss_logging_labels` + wandb.
- **Codec video backend (`video_backend="codec"`) is NOT used at training time.** Justification:
  1. `LMMDataset.preprocess_conversation_stream` pre-decodes the full clip with decord and slices it into time chunks (`clip[i:i+streaming_fps_frames]`); each chunk is a `torch.Tensor` of frames. The codec backend's `process_codec_video(video_url, cfg)` requires a video file path (calls `cv2.VideoCapture` internally) and cannot consume frame tensors.
  2. Codec backend packs the *whole video* into N canvases using cross-frame motion-vector grouping. Slicing the video by time before feeding it would defeat the grouping and pollute its on-disk cache (the cache key in `_cache_dir_for` is `video_url + config` with no time-range component).
  3. It additionally requires `cv-preinfer` (PyPI `codec-video-prep`) + `ffmpeg` + flock + ~2 GB cache per video; non-trivial to deploy across DDP workers.
  4. The README positions codec as a long-video *inference* accelerator; training stays on `video_backend="frames"`, which is the default and matches the existing StreamingVLM pipeline.
  Action: we always go through the default frame-sampling video processor with `num_frames` controlled per call. The processor's `video_backend` kwarg is left at its default (`"frames"`); we never set it to `"codec"`.

### S4. Acceptance criteria

1. `uv pip install` of the existing `.venv-sft` environment loads `AutoModelForImageTextToText.from_pretrained(LLaVA-OV-2 path, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype="auto")` without error.
2. `AutoProcessor.from_pretrained(LLaVA-OV-2 path, trust_remote_code=True)` returns a `LlavaOnevision2Processor` instance whose `tokenizer` has `<|im_start|>=151644` / `<|im_end|>=151645`.
3. `LMMDataset.getitem(0)` on a real `train_s12w24_with_seeks.jsonl` row returns a `BatchFeature` with `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `patch_positions`, `labels`. `labels` is `(1, L)` with at least one non `-100` value, and the count of non `-100` matches `(assistant token count summed across rounds)`.
4. A single `model(**inputs)` call on that batch returns a real `loss` scalar tensor of dtype bf16/fp32 with `.requires_grad` true and finite.
5. `torchrun --standalone --nproc_per_node=1 train.py ...` (single GPU smoke run with `max_steps=2`, `gradient_accumulation_steps=1`) completes 2 optimizer steps and writes a `checkpoint-2/` containing `trainer_state.json`.
6. `torchrun --nproc_per_node=8 train.py ...` over 50 steps shows monotonically (mostly) decreasing train loss in wandb logs.

---

## File Structure

We are **modifying** the existing repo. No restructure. Files touched:

- **Create** `scripts/sft_stage_1_llavaov2.sh` — entry script that points at the LLaVA-OV-2 checkpoint, drops `--use_liger_kernel`, leaves everything else identical.
- **Create** `tests/test_llavaov2_sft_smoke.py` — pytest-style smoke covering criteria S4.1–S4.4. Lives outside `streaming_vlm/` to avoid polluting the package.
- **Modify** `train.py` — branch on a CLI flag (or just on `config.model_type == "llava_onevision2"`) to skip the Qwen2.5-VL-specific patches and use `AutoModelForImageTextToText` + the model's own processor.
- **Modify** `streaming_vlm/data/lmm_dataset.py` — add a branch for `LlavaOnevision2Processor`, convert per-chunk video tensors to `list[np.ndarray]` before sending to processor, and ensure the data collator passes `patch_positions` through.
- **Modify** `streaming_vlm/utils/get_qwen_range.py` — only if the sanity test (Task 2) reveals a mismatch; otherwise no change.

Why this split:
- `train.py` change is one "load model" decision — keep it tight, don't split across files.
- `LMMDataset` change is one "build inputs" decision — also concentrated.
- A separate shell script is the only artifact ops needs to invoke; it's cheap and keeps the existing Qwen2.5-VL script untouched.
- The smoke test is the gate; without it the engineer can claim "it ran" without proving anything.

---

## Pre-flight: verify the env and the auto-loaded classes

### Task 0: Smoke-load model + processor in the existing .venv-sft

**Files:**
- Test (ephemeral): `/tmp/llava_ov2_load_check.py`

- [ ] **Step 1: Write the throwaway load script**

```python
# /tmp/llava_ov2_load_check.py
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct"

cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
assert cfg.model_type == "llava_onevision2", cfg.model_type
assert cfg.text_config.model_type == "qwen3", cfg.text_config.model_type
print("config OK:", cfg.architectures, "/", cfg.text_config.model_type)

proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
assert proc.__class__.__name__ == "LlavaOnevision2Processor", proc.__class__.__name__
tok = proc.tokenizer
assert tok.convert_tokens_to_ids("<|im_start|>") == 151644
assert tok.convert_tokens_to_ids("<|im_end|>") == 151645
assert tok.convert_tokens_to_ids("<|vision_start|>") == 151652
assert tok.convert_tokens_to_ids("<|vision_end|>") == 151653
assert tok.convert_tokens_to_ids("<|image_pad|>") == 151655
assert tok.convert_tokens_to_ids("<|video_pad|>") == 151656
print("processor OK:", proc.__class__.__name__)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
).cuda()
print("model OK:", model.__class__.__name__,
      "language=", model.model.language_model.__class__.__name__,
      "visual=", model.model.visual.__class__.__name__)
```

- [ ] **Step 2: Run it**

Run: `/data/v-kaichen/streaming-vlm/.venv-sft/bin/python /tmp/llava_ov2_load_check.py`
Expected: prints `config OK ...`, `processor OK ...`, `model OK LlavaOnevision2ForConditionalGeneration language= Qwen3Model visual= LlavaOnevision2VisionPretrainedModel`. No exceptions, no warnings about un-initialized weights.

- [ ] **Step 3: If it fails because transformers can't recognize `qwen3` text_config**

Should not happen with transformers 5.7.0 (Qwen3 is fully supported). If you still see `KeyError: 'qwen3'`, verify `.venv-sft/bin/python -c "import transformers; print(transformers.__version__)"` prints `5.7.0`. If lower, re-upgrade:
`VIRTUAL_ENV=/data/v-kaichen/streaming-vlm/.venv-sft uv pip install "transformers==5.7.0"`.

- [ ] **Step 4: Cleanup**

Delete `/tmp/llava_ov2_load_check.py`. Do NOT commit.

No git commit for this task. It's a pre-flight gate.

---

## Tokenizer / chat-template sanity

### Task 1: Verify `get_qwen_range` works on this tokenizer's `previous text` segment

**Files:**
- Test (ephemeral): `/tmp/llava_ov2_range_check.py`

- [ ] **Step 1: Write the range probe**

```python
# /tmp/llava_ov2_range_check.py
import torch
from transformers import AutoProcessor

import sys
sys.path.insert(0, "/data/v-kaichen/streaming-vlm")
from streaming_vlm.utils.get_qwen_range import get_qwen_range

MODEL_PATH = "/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct"
proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

conversation = [
    {"role": "previous text", "content": "alpha beta gamma delta epsilon"},
    {"role": "user", "content": [{"type": "text", "text": "Time=0.0-2.0s hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "world"}]},
]
text = proc.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
print("RENDERED:\n", text, "\n---")
ids = proc.tokenizer(text, return_tensors="pt").input_ids
print("ids shape:", ids.shape)

start, end = get_qwen_range(ids, "previous text", 0, contain_lf=True)
print("previous text range:", start, end)
decoded = proc.tokenizer.decode(ids[0, start:end+1])
print("decoded segment:", repr(decoded))
assert "alpha beta gamma delta epsilon" in decoded, decoded

start, end = get_qwen_range(ids, "assistant", 0, contain_lf=True)
print("assistant range:", start, end)
decoded = proc.tokenizer.decode(ids[0, start:end+1])
print("decoded segment:", repr(decoded))
assert "world" in decoded, decoded
print("range probe OK")
```

- [ ] **Step 2: Run it**

Run: `/data/v-kaichen/streaming-vlm/.venv-sft/bin/python /tmp/llava_ov2_range_check.py`
Expected: prints the rendered prompt (with a leading auto-injected `<|im_start|>system\nYou are a helpful assistant.<|im_end|>` block), then ranges that decode to segments containing `"alpha beta gamma delta epsilon"` and `"world"`. The final line says `range probe OK`.

- [ ] **Step 3: If `get_qwen_range` returns wrong indices**

Open `streaming_vlm/utils/get_qwen_range.py`. The function works off token-id pattern matching for `<|im_start|>{role_tokens}<|im_end|>`. If the auto-injected leading `system` block confuses the `nth_occurrence` indexing (the user passes index `0` to mean "first occurrence of this role"), confirm by reading the file. If the bug is in role-tokenization assumptions, fix it in place. Do not invent a workaround at the call site.

- [ ] **Step 4: Cleanup**

Delete `/tmp/llava_ov2_range_check.py`.

No git commit for this task either. It's a measurement.

---

## Code changes

### Task 2: Add LLaVA-OV-2 branch to `LMMDataset.__init__` (model_base detection)

**Files:**
- Modify: `streaming_vlm/data/lmm_dataset.py:105-114`

- [ ] **Step 1: Read the existing branching to confirm context**

Run: `sed -n '100,118p' streaming_vlm/data/lmm_dataset.py`
Expected: see the `if 'Qwen2VL' in processor.__class__.__name__ ... elif 'Qwen2_5_VL' ... else raise NotImplementedError` block.

- [ ] **Step 2: Replace that block**

Replace:

```python
        if 'Qwen2VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        elif 'Qwen2_5_VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        else:
            raise NotImplementedError(f"Video preprocessing for {processor.__class__.__name__} is not implemented")
```

with:

```python
        proc_name = processor.__class__.__name__
        if 'Qwen2VL' in proc_name or 'Qwen2_5_VL' in proc_name:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        elif 'LlavaOnevision2' in proc_name:
            # Same Qwen tokenizer family; im_start/im_end/assistant IDs are identical.
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'LlavaOnevision2'
        else:
            raise NotImplementedError(f"Video preprocessing for {proc_name} is not implemented")
```

- [ ] **Step 3: Smoke check that constructor still works**

Run:
```bash
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import sys; sys.path.insert(0, '/data/v-kaichen/streaming-vlm')
from transformers import AutoProcessor
from streaming_vlm.data.lmm_dataset import LMMDataset
p = AutoProcessor.from_pretrained('/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct', trust_remote_code=True)
d = LMMDataset(train_annotation_paths=[], processor=p, text_sink=0, text_sliding_window=0)
print('model_base=', d.model_base, 'im_start=', d.im_start_id, 'im_end=', d.im_end_id, 'assistant=', d.assistant_id)
"
```
Expected:
```
model_base= LlavaOnevision2 im_start= 151644 im_end= 151645 assistant= <some id e.g. 77091>
```

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/lmm_dataset.py
git commit -m "feat(dataset): recognize LlavaOnevision2Processor in LMMDataset"
```

---

### Task 3: Convert per-chunk video tensor to `list[np.ndarray]` before processor call

**Files:**
- Modify: `streaming_vlm/data/lmm_dataset.py` — the section that builds `videos=video_inputs` in `getitem` (around `lmm_dataset.py:280` and `:288`).

Background: `preprocess_conversation_stream` puts a `torch.Tensor` of shape `(T,3,H,W)` (uint8) into each `user.content[*].video`. The Qwen2.5-VL processor accepts that via qwen-vl-utils. `LlavaOnevision2VideoProcessor.__call__` (modelchk video_processing_*.py:520-585) accepts file paths, `list[PIL.Image]`, or `list[np.ndarray]` only — see `video_processing_llava_onevision2.py:501-516`. We convert just before the processor call, conditioned on `self.model_base == 'LlavaOnevision2'`. Conversion: `(T,3,H,W)` torch uint8 → `list[np.ndarray (H,W,3) uint8]`.

- [ ] **Step 1: Add the conversion helper at module top (after imports)**

Insert near the top of `streaming_vlm/data/lmm_dataset.py` (after the existing `import` block, e.g., right after the `logger = logging.get_logger(__name__)` line):

```python
def _video_tensor_to_np_frames(video):
    """Convert a (T,3,H,W) uint8 torch.Tensor of frames to list[np.ndarray (H,W,3) uint8]."""
    import numpy as np
    if isinstance(video, torch.Tensor):
        if video.dtype != torch.uint8:
            video = video.to(torch.uint8)
        # (T,3,H,W) -> (T,H,W,3)
        arr = video.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [arr[i] for i in range(arr.shape[0])]
    return video  # already list[np.ndarray] or list[PIL.Image] etc.
```

- [ ] **Step 2: Locate the processor call site**

Run: `sed -n '278,304p' streaming_vlm/data/lmm_dataset.py`
Expected: you see

```python
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)

        conversation = [{"role": "previous text", "content": previous_text}] + conversation

        if return_text:
            return conversation
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
```

- [ ] **Step 3: Adapt the processor call for LlavaOnevision2**

Replace the `inputs = self.processor(...)` block with:

```python
        if self.model_base == 'LlavaOnevision2' and video_inputs is not None:
            # LlavaOnevision2VideoProcessor accepts file path / list[PIL] / list[np.ndarray] only.
            video_inputs = [_video_tensor_to_np_frames(v) for v in video_inputs]
            # The processor's video path expects one <|video_pad|> per video and one frame list per video.
            # Use num_frames=len(frames) to force exact frame count (no resampling at processor level).
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                num_frames=None,        # do not force a fixed count globally; per-video count is implicit
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
```

- [ ] **Step 4: Hand off `patch_positions` through the collator**

The collator currently just returns `batched_inputs[0]` (line 333-334). It returns a `BatchFeature`, so any key the processor produces is already carried through. No change needed unless the trainer strips unknown kwargs — verify by reading `compute_loss_logging_labels` (`streaming_vlm/utils/patch_trainer.py:27`): it does `model(**inputs)`. `LlavaOnevision2ForConditionalGeneration.forward` declares `patch_positions` (line 1283), so this is fine.

Action: no code change. Just confirm by re-reading.

- [ ] **Step 5: Smoke-run `getitem` against a real sample (single non-streaming branch first)**

This step depends on `DATASET_PATH` being set. If you don't yet have access to `Inf-Stream-Train`, skip to Step 6 and circle back after Task 5.

Otherwise:
```bash
DATASET_PATH=/data/v-kaichen/.../Inf-Stream-Train \
/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "
import sys; sys.path.insert(0, '/data/v-kaichen/streaming-vlm')
import os
os.environ.setdefault('VIDEO_MIN_PIXELS', '78400')
os.environ.setdefault('FPS_MAX_FRAMES', '480')
os.environ.setdefault('VIDEO_MAX_PIXELS', '19267584')
from transformers import AutoProcessor
from streaming_vlm.data.lmm_dataset import LMMDataset
p = AutoProcessor.from_pretrained('/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct', trust_remote_code=True, padding_side='right')
import os
DATA = os.environ['DATASET_PATH']
d = LMMDataset(train_annotation_paths=[f'{DATA}/train_s12w24_with_seeks.jsonl'], processor=p, text_sink=512, text_sliding_window=512)
batch = d[0]
for k, v in batch.items():
    print(k, getattr(v, 'shape', type(v).__name__))
nz = (batch['labels'] != -100).sum().item()
print('labels non -100:', nz)
assert nz > 0
"
```
Expected: keys include `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, `patch_positions`, `labels`. `labels non -100` > 0.

- [ ] **Step 6: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add streaming_vlm/data/lmm_dataset.py
git commit -m "feat(dataset): convert video frame tensor to np.ndarray for LlavaOnevision2 processor"
```

---

### Task 4: Adapt `train.py` to load LLaVA-OV-2 and skip Qwen2.5-VL-only patches

**Files:**
- Modify: `train.py:1-99`

- [ ] **Step 1: Re-read the current top of train.py**

Run: `sed -n '1,99p' train.py`
Expected: matches what is documented in the conversation context.

- [ ] **Step 2: Replace the model-loading block**

Replace the section from the top of the file through line 99 (i.e., everything before `trainer = Trainer(...)`) with:

```python
from types import MethodType
import os

import transformers
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    logging,
)

from dataclasses import asdict

from streaming_vlm.utils.patch_trainer import compute_loss_logging_labels
from models import ModelArguments
from streaming_vlm.data.lmm_dataset import DataArguments, LMMDataset, EvalDataArguments

logger = logging.get_logger(__name__)


def find_resume_checkpoint(run_name: str, output_dir: str):
    parent = os.path.dirname(os.path.abspath(output_dir))
    if not os.path.isdir(parent):
        return None
    run_dirs = sorted(
        [d for d in os.listdir(parent) if d.startswith(run_name)],
        reverse=True,
    )
    for d in run_dirs:
        run_path = os.path.join(parent, d)
        print(f"[resume] Checking directory {run_path}")
        if not os.path.isdir(run_path):
            continue
        ckpts = []
        for name in os.listdir(run_path):
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except Exception:
                    step = -1
                ckpts.append((step, os.path.join(run_path, name)))
        ckpts.sort(key=lambda x: x[0], reverse=True)
        for _, cp in ckpts:
            if os.path.isfile(os.path.join(cp, "trainer_state.json")):
                print(f"[resume] Resuming from {cp}")
                return cp
    print("[resume] No checkpoint found")
    return None


def _is_llava_onevision2(config) -> bool:
    return getattr(config, "model_type", None) == "llava_onevision2"


def _is_qwen2_5_vl(config) -> bool:
    archs = getattr(config, "architectures", None) or []
    return any("Qwen2_5_VL" in a or "Qwen2VL" in a for a in archs)


if __name__ == "__main__":
    training_args, model_args, data_args, eval_data_args = HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments, EvalDataArguments)
    ).parse_args_into_dataclasses()

    resume_ckpt = find_resume_checkpoint(training_args.run_name, training_args.output_dir)

    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name_or_path, trust_remote_code=True
    )

    if _is_llava_onevision2(config):
        # Pure-Qwen3 backbone + custom OneVision vision tower. No mRoPE, no rope_deltas,
        # no liger fused linear CE patch. Load via AutoModelForImageTextToText so the
        # `auto_map.AutoModelForImageTextToText` entry resolves the right class.
        model = AutoModelForImageTextToText.from_pretrained(
            model_args.pretrained_model_name_or_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_args.pretrained_model_name_or_path,
            padding_side="right",
            trust_remote_code=True,
        )

        # Freeze the vision tower. For LlavaOnevision2, `model.visual` is a property that
        # returns model.model.visual, so this freezes the actual parameters.
        if hasattr(model, "visual"):
            model.visual.requires_grad_(False)
            print("Freezing module visual (via model.visual property)")
    elif _is_qwen2_5_vl(config):
        # Re-apply the original Qwen2.5-VL hacks only when actually training Qwen2.5-VL.
        import liger_kernel.transformers.model.qwen2_5_vl as qwen2_5_vl
        from streaming_vlm.utils.patch_liger_kernel import lce_forward
        qwen2_5_vl.lce_forward = lce_forward
        from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index

        model = getattr(transformers, config.architectures[0]).from_pretrained(
            model_args.pretrained_model_name_or_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
        )
        model.get_rope_index = MethodType(get_rope_index, model)
        for m in ["visual", "vision_tower"]:
            try:
                getattr(model, m).requires_grad_(False)
                print(f"Freezing module {m}")
            except Exception:
                print(f"Module {m} not found in model")
        if "Qwen2VL" in model.config.architectures[0]:
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", padding_side="right"
            )
        else:
            processor = AutoProcessor.from_pretrained(
                model_args.pretrained_model_name_or_path,
                padding_side="right",
                trust_remote_code=True,
            )
        # Qwen2.5-VL-specific embedding-aliasing hack.
        if hasattr(model, "llm_model_embed_tokens"):
            print("delattr llm_model_embed_tokens")
            delattr(model, "llm_model_embed_tokens")
        setattr(
            type(model),
            "llm_model_embed_tokens",
            property(lambda self: self.llm.model.embed_tokens),
        )
    else:
        raise NotImplementedError(f"Unsupported model_type / arch: {config.model_type} / {getattr(config, 'architectures', None)}")

    train_dataset = LMMDataset(
        **asdict(data_args),
        **asdict(training_args),
        **asdict(model_args),
        processor=processor,
    )
    eval_dataset = LMMDataset(
        **asdict(data_args),
        **asdict(eval_data_args),
        **asdict(training_args),
        **asdict(model_args),
        processor=processor,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.data_collator,
        processing_class=processor,
    )
    trainer.compute_loss = MethodType(compute_loss_logging_labels, trainer)
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else False)
```

- [ ] **Step 3: Verify the file parses**

Run: `/data/v-kaichen/streaming-vlm/.venv-sft/bin/python -c "import ast; ast.parse(open('train.py').read()); print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add train.py
git commit -m "feat(train): branch on model_type to support LLaVA-OneVision-2 SFT"
```

---

### Task 5: Add the LLaVA-OV-2 stage 1 launch script

**Files:**
- Create: `scripts/sft_stage_1_llavaov2.sh`

- [ ] **Step 1: Create the script**

Write `scripts/sft_stage_1_llavaov2.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Pixel + frame budget. OneVision's vision tower uses patch_size=14 (vs Qwen2.5-VL's 14*2=28),
# so the original VIDEO_*_PIXELS budgets carry over to "visual tokens after spatial_merge",
# but the per-frame raw pixel count is half on each axis. We start with the same numbers as
# the Qwen2.5-VL script and tune only if OOM or token-budget issues appear.
export VIDEO_MIN_PIXELS=78400        # ~ 100 visual tokens / frame (post-merge)
export FPS_MAX_FRAMES=480            # 4 min @ 2 FPS
export VIDEO_MAX_PIXELS=19267584     # ~ 24576 visual tokens total per clip

text_sink=512
TEXT_SLIDING_WINDOW=512

: "${DATASET_PATH:?DATASET_PATH must be set to your Inf-Stream-Train directory}"

epoch_num=1
gradient_accumulation_steps=64
learning_rate=1e-5
model_name="/data/v-kaichen/azure_blob/pretrained_models/huggingface/LLaVA-OneVision-2-8B-Instruct"

WANDB_API_KEY=${WANDB_API_KEY:-your-wandb-api-key}
WANDB_ENTITY=${WANDB_ENTITY:-your-wandb-entity}
WANDB_PROJECT_NAME=${WANDB_PROJECT_NAME:-StreamingVLM_LlavaOV2_SFT_stage_1}

timestamp=$(date +%Y%m%d_%H%M%S)

export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}"
export OUTPUT_DIR="./checkpoints"

TRAIN_DATASET_NAMES=(
    "train_s12w24_with_seeks.jsonl"
    "train_s12w24_with_seeks.jsonl"
    "train_livecc_with_seeks.jsonl"
)
VALID_DATASET_NAMES=(
    "valid_s12w24_with_seeks.jsonl"
    "valid_s12w24_with_seeks.jsonl"
    "valid_livecc_with_seeks.jsonl"
)

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$DATASET_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$DATASET_PATH/}")

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_API_KEY=$WANDB_API_KEY \
WANDB_ENTITY=$WANDB_ENTITY \
WANDB_PROJECT=$WANDB_PROJECT_NAME \
TOKENIZERS_PARALLELISM=false \
torchrun --standalone --nproc_per_node=8 train.py \
    --deepspeed ./scripts/zero3.json \
    --overwrite_output_dir True \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}_${timestamp}" \
    --run_name "$RUN_NAME" \
    --save_on_each_node True \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs $epoch_num \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path "$model_name" \
    --train_annotation_paths "${TRAIN_FILES[@]}" \
    --dataloader_num_workers 32 \
    --report_to wandb \
    --ignore_data_skip False \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 10 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --eval_steps 100 \
    --metric_for_best_model eval_loss \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${VALID_FILES[@]}" \
    --text_sink $text_sink \
    --text_sliding_window $TEXT_SLIDING_WINDOW
```

Note the difference vs `sft_stage_1.sh`:
- `model_name` points at the local LLaVA-OV-2 checkpoint.
- `--use_liger_kernel` is removed.
- `WANDB_PROJECT_NAME` is updated.

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/sft_stage_1_llavaov2.sh`

- [ ] **Step 3: Lint with bash -n**

Run: `bash -n scripts/sft_stage_1_llavaov2.sh && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 4: Commit**

```bash
cd /data/v-kaichen/streaming-vlm
git add scripts/sft_stage_1_llavaov2.sh
git commit -m "feat(scripts): add Stage-1 SFT launch script for LLaVA-OneVision-2"
```

---

## Smoke test gate

### Task 6: Single-GPU smoke run, 2 steps

This is the integration gate. It exercises model load + dataset + forward + backward end-to-end.

**Files:** none new.

- [ ] **Step 1: Verify DATASET_PATH is set**

Run:
```bash
export DATASET_PATH=/data/v-kaichen/streaming-vlm/data/Inf-Stream-Train
test -d "$DATASET_PATH/Livecc_sft" && test -f "$DATASET_PATH/train_s12w24_with_seeks.jsonl" && echo "ok" || echo "MISSING"
```
Expected: `ok`.

- [ ] **Step 2: Run a 2-step smoke**

Run:
```bash
cd /data/v-kaichen/streaming-vlm
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
VIDEO_MIN_PIXELS=78400 \
FPS_MAX_FRAMES=480 \
VIDEO_MAX_PIXELS=19267584 \
torchrun --standalone --nproc_per_node=1 train.py \
    --deepspeed ./scripts/zero3.json \
    --overwrite_output_dir True \
    --output_dir ./checkpoints/llavaov2_smoke \
    --run_name llavaov2_smoke \
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
    --train_annotation_paths "$DATASET_PATH/train_s12w24_with_seeks.jsonl" \
    --dataloader_num_workers 0 \
    --report_to none \
    --save_strategy steps \
    --save_steps 2 \
    --save_total_limit 1 \
    --prediction_loss_only true \
    --text_sink 512 \
    --text_sliding_window 512
```
Expected: 2 training steps complete, no NaN loss, a `checkpoints/llavaov2_smoke_*/checkpoint-2/` directory exists with `trainer_state.json`. Loss should be a finite positive number (likely between 0.5 and 3.0 for the first few steps of an instruction-tuned model on streaming data).

- [ ] **Step 3: If OOM on 80GB**

Drop `VIDEO_MAX_PIXELS` to `9633792` (12k visual tokens). If still OOM, drop `FPS_MAX_FRAMES` to `240`. Document the working values in `scripts/sft_stage_1_llavaov2.sh` later.

- [ ] **Step 4: If `KeyError` / unrecognized kwarg in forward**

Most likely cause: data collator stripped `patch_positions`. Re-check `LMMDataset.data_collator` (it just returns `batched_inputs[0]`) and that the `BatchFeature` contains the key. Insert a temporary print right before `model(**inputs)` in `compute_loss_logging_labels` to dump `list(inputs.keys())`.

- [ ] **Step 5: If `tokenizer` complains about `padding_side`**

LLaVA-OV-2's processor stores `padding_side` on the tokenizer; we pass `padding_side='right'` via `AutoProcessor.from_pretrained(...)`. If the warning is just informational, ignore. If it errors, set `processor.tokenizer.padding_side = 'right'` immediately after loading.

- [ ] **Step 6: Commit any fixes**

If you ended up making code edits to clear OOM or kwarg issues, commit them:

```bash
cd /data/v-kaichen/streaming-vlm
git add -p   # review hunks
git commit -m "fix(train): <describe the fix>"
```

If you did NOT need to make any code edits, do not create an empty commit.

---

### Task 7: Multi-GPU 50-step run, confirm loss curve

**Files:** none new.

- [ ] **Step 1: Run 50 steps on 8 GPUs**

Run:
```bash
cd /data/v-kaichen/streaming-vlm
DATASET_PATH=$DATASET_PATH \
WANDB_PROJECT_NAME=StreamingVLM_LlavaOV2_SFT_smoke \
./scripts/sft_stage_1_llavaov2.sh
# After ~50 steps (≈10-20 min depending on data), Ctrl+C.
```
Expected:
- All 8 ranks start.
- Wandb logs `loss` per step.
- Train loss at step ~50 is lower than at step ~5 (allow noise, but trend should be clear).
- `nvidia-smi` shows ~70-90% GPU mem use per device; no OOM.

- [ ] **Step 2: If loss is flat or rises**

Possible causes:
1. Bad `learning_rate` for this backbone. Try `5e-6`.
2. Vision tower not actually frozen (check `print` output at training start for `Freezing module visual ...`).
3. Labels all `-100`. Re-run Task 3 Step 5 with a different sample index to confirm.

- [ ] **Step 3: Commit the verified script**

If you tweaked the script during this task, commit it:

```bash
cd /data/v-kaichen/streaming-vlm
git add scripts/sft_stage_1_llavaov2.sh
git commit -m "fix(scripts): tune Stage-1 LlavaOV2 SFT script after smoke"
```

If unchanged, no commit.

---

## Out of scope (explicitly)

- Stage-2 annealing script: trivially copy `scripts/sft_stage_1_llavaov2.sh` → `sft_stage_2_llavaov2.sh` and change `model_name` + dataset list. Same plan applies. Not detailed here.
- Inference adaptation: requires a new `convert_llava_onevision2_to_streaming` module mirroring `streaming_vlm/inference/qwen2_5/patch_model.py` but targeting Qwen3 attention. Out of scope for this plan.
- Eval scripts (`eval_*.sh`): unchanged. They run inference under `streamingvlm-infer`, which is not yet adapted to LLaVA-OV-2.

---

## Self-review checklist (engineer-side, do this before claiming done)

1. **Spec coverage:** Walk through S4.1–S4.6. Map each to: S4.1 = Task 0 step 2; S4.2 = Task 0 step 2 (processor class) + Task 1 (token ids); S4.3 = Task 3 step 5; S4.4 = Task 6 step 2; S4.5 = Task 6 step 2; S4.6 = Task 7 step 1. All present.
2. **Placeholder scan:** No `TBD`, no `handle X`, every code block self-contained.
3. **Type / name consistency:** `model_base == 'LlavaOnevision2'` is set in Task 2 and checked in Task 3 — names match. `_video_tensor_to_np_frames` is defined and only called in Task 3. `AutoModelForImageTextToText` is used consistently in Task 0 and Task 4.
