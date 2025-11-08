import os, json, math, time, logging
from typing import Optional, Dict, Any, Iterable
from dataclasses import asdict

import torch
from peft import PeftModel
from transformers import TrainingArguments, AutoTokenizer, Trainer
from transformers.trainer_utils import IntervalStrategy
from transformers import default_data_collator

from .config import TrainConfig
from .data import load_any_dataset
from .utils import seed_everything, prepare_model, save_artifacts, maybe_merge_lora
from .logging_utils import setup_logging


def build_tokenizer(model_name: str, max_len: int):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.model_max_length = max_len
    return tok

def tokenize_function(tok, prompt_template: Optional[str]):
    def _fn(batch):
        texts = []
        if "prompt" in batch and "completion" in batch:
            for p, c in zip(batch["prompt"], batch["completion"]):
                if prompt_template:
                    # Ensure the template has a placeholder for the instruction/prompt
                    if "{instruction}" in prompt_template:
                        p = prompt_template.format(instruction=p)
                    else:
                        # Fallback for templates that might just be prefixes
                        p = prompt_template + p
                # Supervised fine-tuning with full target labels
                texts.append(p + "\n" + c)
        elif "text" in batch:
            texts = batch["text"]
        else:
            raise ValueError("Dataset fields must include `prompt` & `completion`, or a combined `text`.")
        
        enc = tok(texts, truncation=True, max_length=tok.model_max_length)
        enc["labels"] = enc["input_ids"].copy()
        return enc
    return _fn

def train(config: TrainConfig, progress_cb=None) -> Dict[str, Any]:
    """
    progress_cb(dict) is called with updates that the API will publish on websockets.
    """
    t0 = time.time()
    setup_logging()
    seed_everything(config.seed)

    # --- Start: Device-aware Configuration ---
    is_cuda_available = torch.cuda.is_available()

    # 1. Conditionally set mixed precision
    use_bf16 = False
    use_fp16 = False
    if config.mixed_precision == "bf16":
        if is_cuda_available and torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            logging.warning("bf16 mixed precision is not supported on this device. Falling back to FP32.")
    elif config.mixed_precision == "fp16":
        if is_cuda_available:
            use_fp16 = True
        else:
            logging.warning("fp16 mixed precision is not supported on this device. Falling back to FP32.")

    # 2. Conditionally set 4-bit loading
    load_in_4bit_safe = config.load_in_4bit
    if load_in_4bit_safe and not is_cuda_available:
        logging.warning("4-bit quantization (bitsandbytes) requires a CUDA GPU. Disabling 'load_in_4bit'.")
        load_in_4bit_safe = False
    # --- End: Device-aware Configuration ---


    if progress_cb: progress_cb({"stage":"loading-dataset"})
    ds = load_any_dataset(config.dataset, field_map=config.dataset_field_map)

    tok = build_tokenizer(config.base_model, config.max_seq_len)

    if progress_cb: progress_cb({"stage":"tokenizing"})
    ds_tok = ds.map(tokenize_function(tok, config.prompt_template), batched=True, remove_columns=ds["train"].column_names)

    if progress_cb: progress_cb({"stage":"loading-model"})
    model = prepare_model(
        config.base_model,
        load_in_4bit=load_in_4bit_safe,  # Use the safe value
        lora_cfg=config.lora
    )

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs if config.max_steps < 0 else 1,
        max_steps=config.max_steps,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_schedule,
        logging_steps=config.logging_steps,
        evaluation_strategy=IntervalStrategy.NO if not config.eval_split else IntervalStrategy.STEPS,
        eval_steps=max(config.logging_steps, 50) if config.eval_split else None,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=use_bf16,  # Use the safe value
        fp16=use_fp16,  # Use the safe value
        gradient_checkpointing=True,
        report_to=["none"],
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok.get(config.eval_split) if config.eval_split else None,
        tokenizer=tok,
        data_collator=default_data_collator
    )

    if progress_cb: progress_cb({"stage":"training-start"})
    trainer.train()
    if progress_cb: progress_cb({"stage":"training-end"})

    # Save adapter + merged fp16/bf16 model (optional)
    if progress_cb: progress_cb({"stage":"saving"})
    paths = save_artifacts(trainer, output_dir)

    # Optional: merge LoRA into base weights to produce a single model
    if isinstance(model, PeftModel):
        merged_path = maybe_merge_lora(model, tok, output_dir, merge=True)
        if merged_path: paths["merged_model_dir"] = merged_path

    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    elapsed = time.time() - t0

    result = {
        "run_id": config.run_id,
        "output": paths,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "config": asdict(config),
    }
    if progress_cb: progress_cb({"stage":"done", "result": result})
    return result