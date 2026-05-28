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
            dtype="auto",
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
            dtype="auto",
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
