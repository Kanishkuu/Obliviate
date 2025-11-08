# backend/trainer/train.py
import os, time, torch
from typing import Dict, Any, Optional
from dataclasses import asdict

from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer
from trl import SFTTrainer

from .config import TrainConfig
from ..services.progress_service import publish
from .logging_utils import setup_logging
from huggingface_hub import HfApi, HfFolder

# ---------- Dataset helpers (ShareGPT-style or text) ----------

def _load_any_dataset_for_unsloth(dataset_name_or_path: str) -> Dataset:
    """Loads HF hub or local json/jsonl/csv into a single split Dataset (train)."""
    if os.path.isdir(dataset_name_or_path) or os.path.isfile(dataset_name_or_path):
        path = dataset_name_or_path.lower()
        if path.endswith(".json") or path.endswith(".jsonl"):
            ds = load_dataset("json", data_files={"train": dataset_name_or_path})["train"]
        elif path.endswith(".csv"):
            ds = load_dataset("csv", data_files={"train": dataset_name_or_path})["train"]
        else:
            raise ValueError("Unsupported local dataset format. Use .json/.jsonl/.csv")
    else:
        # Try HF hub default split
        ds = load_dataset(dataset_name_or_path, split="train")
    return ds

def _format_with_chat_template(tokenizer, ds, chat_template: str, system_message: Optional[str] = None):
    """Standardize + map to text using tokenizer.apply_chat_template on 'conversations'."""
    ds = standardize_sharegpt(ds)
    def _fmt(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}
    # some datasets might be big — keep it single-proc by default for tiny tests
    ds = ds.map(_fmt, batched=True, num_proc=None, remove_columns=ds.column_names if "text" not in ds.column_names else [])
    return ds

# ---------- Main training ----------

def train(config: TrainConfig, progress_cb=None) -> Dict[str, Any]:
    """
    Unsloth + TRL SFTTrainer pipeline that mirrors the user's class.
    Assumes ShareGPT-like dataset (with 'conversations') OR a 'text' column.
    """
    t0 = time.time()
    setup_logging()
    if progress_cb: progress_cb({"stage": "loading-dataset"})

    # 1) Load dataset
    ds = _load_any_dataset_for_unsloth(config.dataset)

    # 2) Load model + tokenizer via Unsloth
    if progress_cb: progress_cb({"stage": "loading-model"})
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_len,
        dtype=dtype,
        load_in_4bit=config.load_in_4bit,
        token=None,  # supply HF token here if needed
    )

    # 3) Chat template onto tokenizer (defaults to Llama-3.1 UX)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
        system_message=None,
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        map_eos_token=False,
    )

    # 4) Build text field:
    #    If dataset has ShareGPT structure, convert to "text" using chat template.
    #    If dataset already has "text", keep it as-is.
    if "text" not in ds.column_names:
        ds = _format_with_chat_template(tokenizer, ds, chat_template="llama-3.1")

    # 5) Apply PEFT (LoRA) via Unsloth
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules or [
            "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
        ],
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=config.lora.use_rslora,
        loftq_config=None,
    )

    # 6) Build trainer (TRL SFTTrainer)
    if progress_cb: progress_cb({"stage": "trainer-setup"})
    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs if config.max_steps < 0 else 1,
        max_steps=config.max_steps,
        learning_rate=config.lr,
        fp16=(not is_bfloat16_supported()),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim=config.optimizer,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        seed=config.seed,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to=["none"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=config.max_seq_len,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=None,  # safe default for small tests
        packing=False,          # could set True for speed w/ short seqs
        args=args,
    )

    # (Optional) Train only on assistant responses
    # trainer = train_on_responses_only(trainer)

    # 7) Train
    if progress_cb: progress_cb({"stage": "training-start"})
    trainer.train()
    if progress_cb: progress_cb({"stage": "training-end"})

    # 8) Save LoRA adapter (and tokenizer) – Unsloth PEFT model behaves like HF PEFT
    if progress_cb: progress_cb({"stage": "saving"})
    os.makedirs(config.output_dir, exist_ok=True)
    adapter_dir = os.path.join(config.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # (Optional) merging LoRA into base weights is non-trivial with Unsloth; skip for minimal path.

    metrics = trainer.state.log_history[-1] if trainer.state.log_history else {}
    result = {
        "run_id": config.run_id,
        "output": {"adapter_dir": adapter_dir},
        "metrics": metrics,
        "elapsed_s": time.time() - t0,
        "config": asdict(config),
    }
    if progress_cb: progress_cb({"stage": "uploading-to-hf"})
    try:
        hf_token = os.getenv("HF_TOKEN")  # set this in .env
        model_name = f"{config.run_id}-{config.base_model.split('/')[-1]}"
        model_dir = os.path.join(config.output_dir, "adapter")
        hf_url = upload_to_huggingface(model_dir, model_name, hf_token)
        result["huggingface_model_url"] = hf_url
        if progress_cb: progress_cb({"stage": "uploaded", "hf_url": hf_url})
    except Exception as e:
        print("⚠️ HF upload failed:", e)

    return result


def upload_to_huggingface(model_dir: str, model_name: str, hf_token: str) -> str:
    """
    Upload fine-tuned model to Hugging Face Hub.
    Returns the model repo URL.
    """
    api = HfApi()
    user = api.whoami(token=hf_token)["name"]
    repo_id = f"{user}/{model_name}"

    print(f"🚀 Uploading model to Hugging Face Hub: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        token=hf_token,
        path_in_repo="",
    )
    return f"https://huggingface.co/{repo_id}"

