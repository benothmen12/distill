"""Supervised fine-tuning with QLoRA (Unsloth + TRL), sized for a free Kaggle T4.

QLoRA keeps the 4B base model frozen in 4-bit and trains small low-rank adapters
on top of every attention and MLP projection: about 1% of the weights, which fits
in 16 GB of GPU memory. The loss is computed on the answer tokens only.

The trainer is built and returned before training, so the notebook can show
exactly which tokens the model learns from before spending GPU time.

GPU libraries are imported inside the functions, so the rest of the project (and
its tests) never needs them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from distill.data.example import Example
from distill.train.format import TemplateMismatch, build_rows, length_report

DEFAULT_BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"


@dataclass
class SFTSettings:
    base_model: str = DEFAULT_BASE_MODEL
    max_length: int = 3072  # system prompt + 12 tool schemas alone are ~2300 tokens
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: float = 2
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation: int = 8  # effective batch of 16 examples
    warmup_ratio: float = 0.05
    seed: int = 42
    output_dir: str = "outputs/sft"
    # Turn markers of the Qwen chat template: the loss starts after `response_part`.
    instruction_part: str = "<|im_start|>user\n"
    response_part: str = "<|im_start|>assistant\n"


def load_model(base_model: str = DEFAULT_BASE_MODEL, max_length: int = 3072) -> tuple[Any, Any]:
    """Load the base model in 4-bit. Float16 on T4/P100, bfloat16 on newer GPUs."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=True,
    )


def make_trainer(
    examples: list[Example], settings: SFTSettings | None = None
) -> tuple[Any, Any, Any, dict]:
    """Load the model, add LoRA adapters and build the trainer, without training."""
    settings = settings or SFTSettings()

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = load_model(settings.base_model, settings.max_length)
    model = FastLanguageModel.get_peft_model(
        model,
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # fmt: skip
        use_gradient_checkpointing="unsloth",
        random_state=settings.seed,
    )

    rows = build_rows(examples, tokenizer)
    lengths = length_report(rows, tokenizer)
    if lengths["max"] > settings.max_length:
        print(f"Warning: {lengths} exceeds max_length={settings.max_length}, some will be cut.")
    for marker in (settings.instruction_part, settings.response_part):
        if marker not in rows[0]["prompt"]:
            raise TemplateMismatch(f"turn marker {marker!r} not found in the chat template")

    try:
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:  # older or newer Unsloth: fall back to TRL's prompt-completion
        train_on_responses_only = None

    if train_on_responses_only is not None:
        data = [{"text": r["prompt"] + r["completion"]} for r in rows]
        loss_mode = "unsloth.train_on_responses_only"
        extra: dict[str, Any] = {"dataset_text_field": "text"}
    else:
        data = rows
        loss_mode = "trl.completion_only_loss"
        extra = {"completion_only_loss": True}
    dataset = Dataset.from_list(data).shuffle(seed=settings.seed)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=settings.output_dir,
            max_length=settings.max_length,
            num_train_epochs=settings.epochs,
            per_device_train_batch_size=settings.batch_size,
            gradient_accumulation_steps=settings.gradient_accumulation,
            learning_rate=settings.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=settings.warmup_ratio,
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=bf16,
            fp16=not bf16,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=settings.seed,
            **extra,
        ),
    )
    if train_on_responses_only is not None:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=settings.instruction_part,
            response_part=settings.response_part,
        )

    info = {
        "settings": asdict(settings),
        "train_examples": len(rows),
        "token_lengths": lengths,
        "loss_mode": loss_mode,
    }
    return model, tokenizer, trainer, info


def trained_tokens(trainer: Any, tokenizer: Any, index: int = 0) -> str:
    """The part of a training example the loss is computed on, decoded.

    It should be the expected answer only: the tool calls or the reply.
    """
    labels = trainer.train_dataset[index]["labels"]
    return tokenizer.decode([t for t in labels if t != -100])


def run_training(model: Any, trainer: Any, info: dict) -> dict:
    """Train, switch the model to inference mode, and return the run info."""
    from unsloth import FastLanguageModel

    result = trainer.train()
    FastLanguageModel.for_inference(model)
    return {
        **info,
        "train_loss": round(result.training_loss, 4),
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "loss_history": [
            {"step": log["step"], "loss": log["loss"]}
            for log in trainer.state.log_history
            if "loss" in log
        ],
    }
