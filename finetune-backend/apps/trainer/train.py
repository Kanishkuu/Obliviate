import os, json, math, time
from typing import Optional, Dict, Any, Iterable
from dataclasses import asdict
from .config import TrainConfig
from .data import load_any_dataset
from .utils import seed_everything, prepare_model, save_artifacts, maybe_merge_lora
from .logging_utils import setup_logging
from transformers import TrainingArguments, AutoTokenizer
from transformers.trainer_utils import IntervalStrategy
from transformers import default_data_collator

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
                    p = prompt_template.format(instruction=p)
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

    if progress_cb: progress_cb({"stage":"loading-dataset"})
    ds = load_any_dataset(config.dataset, field_map=config.dataset_field_map)

    tok = build_tokenizer(config.base_model, config.max_seq_len)

    if progress_cb: progress_cb({"stage":"tokenizing"})
    ds_tok = ds.map(tokenize_function(tok, config.prompt_template), batched=True, remove_columns=ds["train"].column_names)

    if progress_cb: progress_cb({"stage":"loading-model"})
    model = prepare_model(
        config.base_model,
        load_in_4bit=config.load_in_4bit,
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
        bf16=(config.mixed_precision=="bf16"),
        fp16=(config.mixed_precision=="fp16"),
        gradient_checkpointing=True,
        report_to=["none"],
        dataloader_num_workers=2
    )

    from transformers import Trainer
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
