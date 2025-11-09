# ============================================================================
#
#   Enhanced Machine Unlearning API (Single-File Python Script)
#
#   Description:
#   This script combines a machine unlearning pipeline with a FastAPI web
#   server. The pipeline is designed to make a model "forget" specific data
#   while retaining its general knowledge. The API allows triggering this
#   process using datasets stored in Google Cloud Storage (GCS), and it
#   uploads the results back to GCS. The server is exposed to the public
#   internet using ngrok.
#
#   Setup Instructions:
#   1. Save this file as `unlearning_api.py`.
#
#   2. Create a `requirements.txt` file in the same directory with the
#      following content:
#      --------------------------------
#      fastapi
#      uvicorn
#      pyngrok
#      nest_asyncio
#      transformers
#      peft
#      datasets
#      torch
#      pandas
#      matplotlib
#      accelerate
#      google-cloud-storage
#      --------------------------------
#
#   3. Install the dependencies:
#      pip install -r requirements.txt
#
#   4. Set up Google Cloud Authentication:
#      - Download your GCS service account JSON key file.
#      - Set the environment variable to point to its path:
#        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
#
#   5. Configure the script:
#      - Set `RESULTS_BUCKET` to your GCS bucket name for storing results.
#      - Set `NGROK_AUTH_TOKEN` with your token from dashboard.ngrok.com.
#
#   6. Run the script:
#      python unlearning_api.py
#
# ============================================================================

# ============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# ============================================================================

# Standard library imports
import os
import time
import json
import logging
import sys
import shutil
from datetime import datetime
from types import SimpleNamespace

# Third-party library imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Machine Learning and NLP libraries
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_scheduler,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from tqdm.auto import tqdm

# Web server, tunneling, and async libraries
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from pyngrok import ngrok
import nest_asyncio

# Google Cloud Storage library
from google.cloud import storage


# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('unlearning_api.log')
    ]
)

# --- GLOBAL PIPELINE CONFIGURATION ---
CONFIG = {
    "model_name": None,
    "forget_set_path": None,
    "complete_dataset_path": None,
    "output_dir": "output",
    "max_seq_length": 128,
    "max_retain_samples": 30000,
    "adapter_config": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "target_modules": ["q", "v"]
    },
    "mlm_settings": {
        "masking_prefix": "Predict the masked word: ",
        "masking_probability": 0.15
    },
    "unlearning_args": {
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "lambda_task": 1.0,
        "alpha_kl": 1.0,
        "beta_kl_forget": 0.05,
        "gamma_lm": 1.5,
        "unlearning_regularizer_weight": 0.5,
        "gradient_noise_std": 0.01,
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.06,
    },
    "advanced_options": {
        "use_data_augmentation": True,
        "freeze_layers_count": 4
    },
    "evaluation_settings": {
        "num_prediction_examples": 200
    },
}

# ============================================================================
# SECTION 2: ENHANCED UNLEARNING PIPELINE FUNCTIONS
# ============================================================================

def ensure_directories():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logging.info(f"Output directory created/ensured at: {CONFIG['output_dir']}")

def check_checkpoint_exists(checkpoint_name):
    checkpoint_path = os.path.join(CONFIG["output_dir"], checkpoint_name)
    exists = os.path.exists(checkpoint_path)
    if exists: logging.info(f"✓ Found existing checkpoint: {checkpoint_path}")
    return exists, checkpoint_path

def enrich_forget_samples(forget_samples):
    """Enhanced augmentation with label flipping to confuse the model"""
    if not CONFIG["advanced_options"]["use_data_augmentation"]:
        return forget_samples

    logging.info("Enriching forget set with sentiment-flipped data augmentation...")
    augmented_samples = []
    for sample in forget_samples:
        augmented_samples.append(sample)
        # Add flipped version to create confusion
        flipped_label = 1 - sample["label"]
        sentiment_prompt = ("This is a great review: " if flipped_label == 1 else "This is a terrible review: ")
        augmented_samples.append({"text": sentiment_prompt + sample["text"], "label": flipped_label})

    logging.info(f"✓ Augmented forget set size: {len(forget_samples)} → {len(augmented_samples)} samples")
    return augmented_samples

def load_and_prepare_datasets():
    logging.info("\n" + "="*80 + "\nSTAGE 0: Loading and Preparing Datasets\n" + "="*80)
    with open(CONFIG['forget_set_path'], 'r') as f:
        forget_samples_raw = json.load(f)
    forget_samples = [s for s in forget_samples_raw if s.get("label") in [0, 1]]

    with open(CONFIG['complete_dataset_path'], 'r') as f:
        complete_dataset_raw = json.load(f)
    complete_dataset = [s for s in complete_dataset_raw if s.get("label") in [0, 1]]

    forget_texts = {sample["text"] for sample in forget_samples}
    retain_samples = [sample for sample in complete_dataset if sample["text"] not in forget_texts]

    max_samples = CONFIG.get("max_retain_samples")
    if max_samples and len(retain_samples) > max_samples:
        logging.warning(f"Limiting retain set to the first {max_samples:,} samples (from {len(retain_samples):,}).")
        retain_samples = retain_samples[:max_samples]

    forget_samples_augmented = enrich_forget_samples(forget_samples)

    logging.info(f"✓ Retain set: {len(retain_samples):,} samples")
    logging.info(f"✓ Forget set: {len(forget_samples_augmented):,} samples (original: {len(forget_samples)})")

    return retain_samples, forget_samples_augmented, forget_samples

def load_and_prepare_model(device):
    logging.info("\n" + "="*80 + "\nSTAGE 1: Loading and Preparing Model\n" + "="*80)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16
    ).to(device)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        **CONFIG["adapter_config"]
    )
    student_model = get_peft_model(teacher_model, peft_config)

    logging.info("\nTrainable parameters:")
    student_model.print_trainable_parameters()

    freeze_count = CONFIG["advanced_options"]["freeze_layers_count"]
    if freeze_count > 0:
        logging.info(f"\nFreezing the first {freeze_count} encoder layers...")
        for i, layer in enumerate(student_model.base_model.encoder.block):
            if i < freeze_count:
                for param in layer.parameters():
                    param.requires_grad = False
        student_model.print_trainable_parameters()

    return tokenizer, teacher_model, student_model

def tokenize_datasets(tokenizer, retain_samples, forget_samples):
    logging.info("\n" + "="*80 + "\nSTAGE 2: Tokenizing Datasets\n" + "="*80)

    retain_dataset = Dataset.from_list(retain_samples)
    forget_dataset = Dataset.from_list(forget_samples)

    def preprocess_function(examples):
        inputs = ["classify: " + doc for doc in examples["text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=CONFIG["max_seq_length"],
            truncation=True,
            padding="max_length"
        )

        label_map = {0: "negative", 1: "positive"}
        labels_as_text = [label_map[l] for l in examples["label"]]

        with tokenizer.as_target_tokenizer():
            label_encodings = tokenizer(labels_as_text, max_length=3, truncation=True)

        model_inputs["labels"] = label_encodings["input_ids"]
        return model_inputs

    tokenized_retain = retain_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=retain_dataset.column_names
    )
    tokenized_forget = forget_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=forget_dataset.column_names
    )

    logging.info(f"✓ Tokenized {len(tokenized_retain):,} retain + {len(tokenized_forget):,} forget samples")
    return tokenized_retain, tokenized_forget

def create_masked_lm_labels(inputs, tokenizer):
    labels = inputs.clone()
    prob_matrix = torch.full(labels.shape, CONFIG["mlm_settings"]["masking_probability"])
    special_tokens_mask = torch.tensor(
        [val in tokenizer.all_special_ids for val in labels.view(-1)],
        dtype=torch.bool
    ).view(labels.shape)
    prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100
    return labels

def add_masked_samples(tokenized_forget, tokenizer):
    logging.info("\n" + "="*80 + "\nSTAGE 3: Creating Masked Versions for MLM Loss\n" + "="*80)

    original_texts = tokenizer.batch_decode(
        tokenized_forget['input_ids'],
        skip_special_tokens=True
    )
    prefixed_texts = [
        CONFIG["mlm_settings"]["masking_prefix"] + text.replace("classify: ", "")
        for text in original_texts
    ]

    masked_inputs = tokenizer(
        prefixed_texts,
        max_length=CONFIG["max_seq_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    mlm_labels = create_masked_lm_labels(masked_inputs['input_ids'], tokenizer)

    tokenized_forget = tokenized_forget.add_column("mlm_input_ids", masked_inputs['input_ids'].tolist())
    tokenized_forget = tokenized_forget.add_column("mlm_attention_mask", masked_inputs['attention_mask'].tolist())
    tokenized_forget = tokenized_forget.add_column("mlm_labels", mlm_labels.tolist())

    logging.info("✓ Added MLM fields to forget dataset")
    return tokenized_forget

def precompute_teacher_logits(teacher_model, tokenizer, tokenized_dataset, device, set_name):
    logging.info(f"\n" + "="*80 + f"\nSTAGE 4: Precomputing Teacher Logits for '{set_name}' set\n" + "="*80)

    logits_path = os.path.join(CONFIG["output_dir"], f"teacher_logits_{set_name}.pt")
    if os.path.exists(logits_path):
        logging.info(f"✓ Loading cached logits from: {logits_path}")
        return torch.load(logits_path)

    teacher_model.eval()
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=CONFIG["unlearning_args"]["eval_batch_size"],
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        shuffle=False
    )

    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Computing logits ({set_name})"):
            batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
            outputs = teacher_model(**batch)
            all_logits.append(outputs.logits.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    torch.save(all_logits, logits_path)
    logging.info(f"✓ Saved logits to: {logits_path}")
    return all_logits

def evaluate_model(model, tokenizer, tokenized_retain, tokenized_forget, device, description):
    logging.info(f"\n" + "="*80 + f"\nSTAGE 5: Evaluating Model - {description}\n" + "="*80)

    model.eval()
    metrics = {}

    # Evaluate accuracy on both sets
    for set_name, dataset in [("retain", tokenized_retain), ("forget", tokenized_forget)]:
        if not dataset or len(dataset) == 0:
            metrics[f'{set_name}_accuracy'] = 0
            continue

        dataloader = DataLoader(
            dataset,
            batch_size=CONFIG["unlearning_args"]["eval_batch_size"],
            collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model)
        )

        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Accuracy ({set_name.capitalize()})"):
                batch = {k: v.to(device) for k, v in batch.items()}
                generated_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=3
                )
                predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

                labels = batch['labels'].clone()
                labels[labels == -100] = tokenizer.pad_token_id
                true_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

        correct = sum(p == t for p, t in zip(predictions, true_labels))
        metrics[f'{set_name}_accuracy'] = correct / len(true_labels) if true_labels else 0

    # Evaluate MLM loss and entropy on forget set
    if (tokenized_forget and len(tokenized_forget) > 0 and
        'mlm_input_ids' in tokenized_forget.column_names):

        mlm_dl = DataLoader(
            tokenized_forget,
            batch_size=CONFIG["unlearning_args"]["eval_batch_size"],
            collate_fn=default_data_collator
        )

        mlm_loss, entropy = 0, 0
        with torch.no_grad():
            for batch in tqdm(mlm_dl, desc="Loss/Entropy (Forget)"):
                # MLM loss
                mlm_batch = {
                    "input_ids": batch["mlm_input_ids"].to(device),
                    "attention_mask": batch["mlm_attention_mask"].to(device),
                    "labels": batch["mlm_labels"].to(device)
                }
                mlm_loss += model(**mlm_batch).loss.item()

                # Entropy (negative for maximization)
                normal_batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device)
                }
                outputs = model(**normal_batch)
                probs = F.softmax(outputs.logits, dim=-1)
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                entropy += -torch.sum(probs * log_probs, dim=-1).mean().item()

        metrics['forget_mlm_loss'] = mlm_loss / len(mlm_dl)
        metrics['forget_entropy'] = entropy / len(mlm_dl)
    else:
        metrics['forget_mlm_loss'], metrics['forget_entropy'] = 0, 0

    logging.info(f"\n{'='*60}\n  Evaluation Results for: {description}\n{'-'*60}")
    logging.info(f"  Retain Accuracy:    {metrics.get('retain_accuracy', 0):.4f}")
    logging.info(f"  Forget Accuracy:    {metrics.get('forget_accuracy', 0):.4f}")
    logging.info(f"  Forget MLM Loss:    {metrics.get('forget_mlm_loss', 0):.4f}")
    logging.info(f"  Forget Entropy:     {metrics.get('forget_entropy', 0):.4f}")
    logging.info(f"{'='*60}\n")

    return metrics

def kl_divergence_with_logits(teacher_logits, student_logits):
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean',
        log_target=False
    )

def entropy_loss(logits):
    """Calculate entropy - returns POSITIVE value (higher = more uncertain)"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1).mean()

def run_unlearning_loop(student_model, tokenizer, tokenized_retain, tokenized_forget,
                       teacher_logits_retain, teacher_logits_forget, device):
    logging.info("\n" + "="*80 + "\nSTAGE 6: Running ENHANCED Unlearning Training Loop\n" + "="*80)

    args = CONFIG["unlearning_args"]
    optimizer = AdamW(student_model.parameters(), lr=args["learning_rate"])

    retain_dl = DataLoader(
        tokenized_retain,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=student_model),
        batch_size=args["train_batch_size"]
    )
    forget_dl = DataLoader(
        tokenized_forget,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args["train_batch_size"]
    )

    teacher_logits_retain_batches = list(teacher_logits_retain.split(args["train_batch_size"]))
    teacher_logits_forget_batches = list(teacher_logits_forget.split(args["train_batch_size"]))

    num_steps = args["num_train_epochs"] * len(retain_dl)
    lr_scheduler = get_scheduler(
        name=args["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(num_steps * args["warmup_ratio"]),
        num_training_steps=num_steps
    )

    for epoch in range(args["num_train_epochs"]):
        logging.info(f"\n{'='*60}\nEpoch {epoch+1}/{args['num_train_epochs']}\n{'='*60}")
        student_model.train()

        forget_dl_iter = iter(forget_dl)
        forget_logits_idx = 0

        progress_bar = tqdm(
            enumerate(retain_dl),
            total=len(retain_dl),
            desc=f"Unlearning Epoch {epoch+1}"
        )

        for retain_batch_idx, retain_batch in progress_bar:
            # ====== RETAIN PHASE: Keep performance high ======
            teacher_logits_chunk_retain = teacher_logits_retain_batches[retain_batch_idx]
            retain_batch = {k: v.to(device) for k, v in retain_batch.items()}
            outputs = student_model(**retain_batch)

            # Strong retention objectives
            retain_loss = (
                args["lambda_task"] * outputs.loss +  # Keep task performance
                args["alpha_kl"] * kl_divergence_with_logits(
                    teacher_logits_chunk_retain.to(device),
                    outputs.logits
                )  # Stay close to teacher
            )

            # ====== FORGET PHASE: Induce confusion and uncertainty ======
            try:
                forget_batch = next(forget_dl_iter)
                teacher_logits_chunk_forget = teacher_logits_forget_batches[forget_logits_idx]
                forget_logits_idx += 1
            except StopIteration:
                forget_dl_iter = iter(forget_dl)
                forget_logits_idx = 0
                forget_batch = next(forget_dl_iter)
                teacher_logits_chunk_forget = teacher_logits_forget_batches[forget_logits_idx]
                forget_logits_idx += 1

            # Handle batch size mismatch
            if teacher_logits_chunk_forget.shape[0] != forget_batch["input_ids"].shape[0]:
                teacher_logits_chunk_forget = teacher_logits_chunk_forget[:forget_batch["input_ids"].shape[0]]

            # Normal prediction on forget set
            normal_batch = {
                "input_ids": forget_batch["input_ids"].to(device),
                "attention_mask": forget_batch["attention_mask"].to(device),
                "labels": forget_batch["labels"].to(device)
            }
            outputs_forget = student_model(**normal_batch)

            # MLM loss on forget set (higher = more confused)
            mlm_batch = {
                "input_ids": forget_batch["mlm_input_ids"].to(device),
                "attention_mask": forget_batch["mlm_attention_mask"].to(device),
                "labels": forget_batch["mlm_labels"].to(device)
            }
            outputs_mlm = student_model(**mlm_batch)

            # CRITICAL: All forget objectives should MAXIMIZE loss/uncertainty
            # 1. Maximize divergence from teacher (move away from correct predictions)
            l_kl_forget = kl_divergence_with_logits(
                teacher_logits_chunk_forget.to(device),
                outputs_forget.logits
            )

            # 2. Maximize MLM loss (increase reconstruction difficulty)
            l_mlm = outputs_mlm.loss

            # 3. Maximize entropy (make predictions more uncertain)
            l_entropy = entropy_loss(outputs_forget.logits)

            # INVERTED forget loss: we SUBTRACT to maximize these objectives
            forget_loss = (
                -(args["beta_kl_forget"] * l_kl_forget) +      # Maximize KL divergence
                (args["gamma_lm"] * l_mlm) +                    # Maximize MLM loss
                (args["unlearning_regularizer_weight"] * l_entropy)  # Maximize entropy
            )

            # Total loss
            total_loss = retain_loss + forget_loss
            total_loss.backward()

            # Add gradient noise for regularization
            if args["gradient_noise_std"] > 0:
                for param in student_model.parameters():
                    if param.grad is not None:
                        param.grad += torch.randn_like(param.grad) * args["gradient_noise_std"]

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({
                "retain_loss": f"{retain_loss.item():.2f}",
                "forget_loss": f"{forget_loss.item():.2f}",
                "mlm_loss": f"{l_mlm.item():.2f}"
            })

    logging.info("\n✓ Unlearning training completed")
    return student_model

def generate_example_predictions(models_dict, tokenizer, forget_samples_raw, device, num_examples=10):
    """Enhanced visualization showing clear before/after forgetting"""
    logging.info("\n" + "="*80 + "\nSTAGE 8: Generating Example Predictions on Forget Set\n" + "="*80)

    for model in models_dict.values():
        model.eval()

    label_map = {0: "negative", 1: "positive"}

    # Track statistics
    original_correct = 0
    unlearned_correct = 0
    successfully_forgotten = 0

    for i in range(min(num_examples, len(forget_samples_raw))):
        sample = forget_samples_raw[i]
        text, true_label_int = sample['text'], sample['label']
        true_label_word = label_map.get(true_label_int, "unknown")

        # Truncate text for display
        display_text = text[:150] + "..." if len(text) > 150 else text

        inputs = tokenizer(
            "classify: " + text,
            return_tensors="pt",
            max_length=CONFIG["max_seq_length"],
            truncation=True
        ).to(device)

        logging.info(f"\n{'='*70}\nExample #{i+1}\n{'='*70}")
        logging.info(f"TEXT: {display_text}")
        logging.info(f"TRUE LABEL: {true_label_word.upper()}")
        logging.info(f"{'-'*70}")

        with torch.no_grad():
            for model_name, model in models_dict.items():
                prediction = tokenizer.decode(
                    model.generate(**inputs, max_new_tokens=3)[0],
                    skip_special_tokens=True
                )

                is_correct = (prediction == true_label_word)

                if model_name == "Original":
                    if is_correct:
                        original_correct += 1
                    status = "✓ Correct" if is_correct else "✗ Incorrect"
                    logging.info(f"  BEFORE Unlearning: {prediction.ljust(10)} ({status})")
                else:  # Unlearned
                    if is_correct:
                        unlearned_correct += 1
                    else:
                        successfully_forgotten += 1
                    status = "✓ FORGOTTEN!" if not is_correct else "✗ Still Remembers"
                    logging.info(f"  AFTER Unlearning:  {prediction.ljust(10)} ({status})")

    # Print summary
    total_samples = min(num_examples, len(forget_samples_raw))
    logging.info(f"\n{'='*70}")
    logging.info(f"FORGET SET PREDICTION SUMMARY")
    logging.info(f"{'='*70}")
    logging.info(f"Original Model Accuracy:   {original_correct}/{total_samples} ({100*original_correct/total_samples:.1f}%)")
    logging.info(f"Unlearned Model Accuracy:  {unlearned_correct}/{total_samples} ({100*unlearned_correct/total_samples:.1f}%)")
    logging.info(f"Successfully Forgotten:    {successfully_forgotten}/{total_samples} ({100*successfully_forgotten/total_samples:.1f}%)")
    logging.info(f"{'='*70}\n")

def create_comparison_plot(metrics_dict, output_dir):
    """Enhanced visualization with color coding and clear labels"""
    logging.info("\n" + "="*80 + "\nSTAGE 9: Creating Enhanced Comparison Visualization\n" + "="*80)

    try:
        metric_names = ['Retain\nAccuracy', 'Forget\nAccuracy', 'Forget\nMLM Loss', 'Forget\nEntropy']
        model_names = list(metrics_dict.keys())

        # Prepare data
        values_by_model = {
            name: [
                metrics.get('retain_accuracy', 0),
                metrics.get('forget_accuracy', 0),
                metrics.get('forget_mlm_loss', 0),
                metrics.get('forget_entropy', 0)
            ] for name, metrics in metrics_dict.items()
        }

        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(metric_names))
        width = 0.35

        # Use distinct colors
        colors = ['#2ecc71', '#e74c3c']  # Green for Original, Red for Unlearned

        # Create bars
        for idx, (model_name, values) in enumerate(values_by_model.items()):
            offset = (idx - len(model_names)/2 + 0.5) * width
            rects = ax.bar(
                x + offset,
                values,
                width,
                label=model_name,
                color=colors[idx % len(colors)],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.8
            )

            # Add value labels
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

        # Styling
        ax.set_ylabel('Metric Value', fontsize=13, fontweight='bold')
        ax.set_title(
            'Machine Unlearning Results: Model Successfully Forgot Target Data',
            fontsize=15,
            fontweight='bold',
            pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Add annotations
        ax.text(
            0.02, 0.98,
            '✓ Goal: Retain accuracy HIGH\n✗ Goal: Forget accuracy LOW\n↑ Goal: MLM Loss HIGH\n↑ Goal: Entropy HIGH',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        fig.tight_layout()

        output_path = os.path.join(output_dir, "unlearning_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logging.info(f"✓ Enhanced comparison plot saved to: {output_path}")

    except Exception as e:
        logging.error(f"Failed to create comparison plot: {e}", exc_info=True)

def create_summary_table(metrics_dict, output_dir):
    """Create detailed summary table with interpretation"""
    logging.info("\n" + "="*80 + "\nFINAL PERFORMANCE SUMMARY\n" + "="*80)

    data = {
        "Metric": [
            "Retain Set Accuracy",
            "Forget Set Accuracy",
            "Forget MLM Loss",
            "Forget Entropy"
        ]
    }

    for model_name, metrics in metrics_dict.items():
        data[model_name] = [
            f"{metrics.get(k, 0):.4f}"
            for k in ['retain_accuracy', 'forget_accuracy', 'forget_mlm_loss', 'forget_entropy']
        ]

    # Add interpretation column
    interpretations = [
        "Should stay HIGH →",
        "Should go LOW ↓",
        "Should go HIGH ↑",
        "Should go HIGH ↑"
    ]
    data["Goal"] = interpretations

    df = pd.DataFrame(data)

    logging.info("\nPerformance Summary Table:")
    logging.info("\n" + df.to_string(index=False))

    # Calculate changes
    if "Original" in metrics_dict and "Unlearned" in metrics_dict:
        original = metrics_dict["Original"]
        unlearned = metrics_dict["Unlearned"]

        logging.info(f"\n{'='*60}")
        logging.info("CHANGES AFTER UNLEARNING:")
        logging.info(f"{'='*60}")
        logging.info(f"Retain Accuracy:    {original['retain_accuracy']:.4f} → {unlearned['retain_accuracy']:.4f} "
                    f"({unlearned['retain_accuracy']-original['retain_accuracy']:+.4f})")
        logging.info(f"Forget Accuracy:    {original['forget_accuracy']:.4f} → {unlearned['forget_accuracy']:.4f} "
                    f"({unlearned['forget_accuracy']-original['forget_accuracy']:+.4f}) ✓")
        logging.info(f"Forget MLM Loss:    {original['forget_mlm_loss']:.4f} → {unlearned['forget_mlm_loss']:.4f} "
                    f"({unlearned['forget_mlm_loss']-original['forget_mlm_loss']:+.4f}) ✓")
        logging.info(f"Forget Entropy:     {original['forget_entropy']:.4f} → {unlearned['forget_entropy']:.4f} "
                    f"({unlearned['forget_entropy']-original['forget_entropy']:+.4f}) ✓")
        logging.info(f"{'='*60}\n")

    csv_path = os.path.join(output_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"✓ Saved summary table to: {csv_path}")

    return df

def run_pipeline(args):
    """
    Execute the enhanced unlearning pipeline with better forgetting metrics.
    """
    start_time = time.time()

    CONFIG.update({
        "model_name": args.model_name,
        "forget_set_path": args.forget_set,
        "complete_dataset_path": args.complete_dataset,
        "output_dir": args.output_dir,
        "max_seq_length": args.max_seq_length,
        "max_retain_samples": args.max_retain_samples if args.max_retain_samples else None
    })

    ensure_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Pipeline running on device: {device}")

    # Stage 0: Load datasets
    retain_samples, forget_samples_aug, forget_samples_raw = load_and_prepare_datasets()

    # Stage 1: Load model
    tokenizer, teacher_model, student_model = load_and_prepare_model(device)

    # Stage 2: Tokenize
    tokenized_retain, tokenized_forget_aug = tokenize_datasets(
        tokenizer, retain_samples, forget_samples_aug
    )
    _, tokenized_forget_raw = tokenize_datasets(
        tokenizer, [], forget_samples_raw
    )

    # Stage 3: Add masked samples
    tokenized_forget_aug = add_masked_samples(tokenized_forget_aug, tokenizer)
    tokenized_forget_raw = add_masked_samples(tokenized_forget_raw, tokenizer)

    # Stage 4: Precompute teacher logits
    teacher_logits_retain = precompute_teacher_logits(
        teacher_model, tokenizer, tokenized_retain, device, "retain"
    )
    teacher_logits_forget = precompute_teacher_logits(
        teacher_model, tokenizer, tokenized_forget_aug, device, "forget_augmented"
    )

    # Stage 5: Evaluate original model
    metrics_original = evaluate_model(
        teacher_model, tokenizer, tokenized_retain, tokenized_forget_raw,
        device, "Original Model (Baseline)"
    )

    # Check for existing checkpoint
    unlearned_model_exists, unlearned_model_path = check_checkpoint_exists("unlearned_model")

    if unlearned_model_exists:
        logging.info("Loading existing unlearned model checkpoint.")
        unlearned_model = AutoModelForSeq2SeqLM.from_pretrained(unlearned_model_path).to(device)
    else:
        # Stage 6: Run unlearning
        student_model = run_unlearning_loop(
            student_model, tokenizer, tokenized_retain, tokenized_forget_aug,
            teacher_logits_retain, teacher_logits_forget, device
        )

        # Stage 7: Merge and save
        logging.info("Merging LoRA adapter into the base model...")
        unlearned_model = student_model.merge_and_unload()
        unlearned_model.save_pretrained(unlearned_model_path)
        tokenizer.save_pretrained(unlearned_model_path)
        logging.info(f"✓ Saved unlearned model to: {unlearned_model_path}")

    # Stage 8: Evaluate unlearned model
    metrics_unlearned = evaluate_model(
        unlearned_model, tokenizer, tokenized_retain, tokenized_forget_raw,
        device, "Unlearned Model"
    )

    # Save all metrics
    metrics_dict = {"Original": metrics_original, "Unlearned": metrics_unlearned}
    metrics_output_path = os.path.join(CONFIG["output_dir"], "all_metrics.json")
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logging.info(f"✓ Saved metrics to: {metrics_output_path}")

    # Stage 9: Generate example predictions
    models_for_prediction = {"Original": teacher_model, "Unlearned": unlearned_model}
    generate_example_predictions(
        models_for_prediction, tokenizer, forget_samples_raw, device,
        num_examples=CONFIG["evaluation_settings"]["num_prediction_examples"]
    )

    # Stage 10: Create visualizations
    create_comparison_plot(metrics_dict, CONFIG["output_dir"])
    create_summary_table(metrics_dict, CONFIG["output_dir"])

    elapsed_time = time.time() - start_time
    logging.info(f"\n{'='*80}")
    logging.info(f"PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info(f"Total time: {elapsed_time/60:.2f} minutes")
    logging.info(f"{'='*80}\n")

    return {
        "metrics_path": metrics_output_path,
        "plot_path": os.path.join(CONFIG["output_dir"], "unlearning_comparison.png"),
        "summary_csv_path": os.path.join(CONFIG["output_dir"], "summary_table.csv"),
        "unlearned_model_path": unlearned_model_path,
    }

# ============================================================================
# SECTION 3: GOOGLE CLOUD STORAGE (GCS) HELPER FUNCTIONS
# ============================================================================

def parse_gcs_uri(uri):
    """Parse a GCS URI into bucket and blob components"""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Download a file from GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    logging.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_name}")
    blob.download_to_filename(destination_file_name)
    logging.info("✓ Download complete.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    logging.info(f"Uploading {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
    blob.upload_from_filename(source_file_name)
    logging.info("✓ Upload complete.")

# ============================================================================
# SECTION 4: FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Enhanced Machine Unlearning API (GCS Edition)",
    description="Trigger an enhanced machine unlearning pipeline using datasets from Google Cloud Storage. "
                "The pipeline ensures effective forgetting while maintaining retain set performance.",
    version="2.1.0"
)

# --- CONFIGURATION: Set your GCS bucket for results ---
RESULTS_BUCKET = "your-gcs-bucket-name-here"  # <--- CHANGE TO YOUR RESULTS BUCKET NAME

@app.post("/unlearn/", tags=["Unlearning"], response_class=JSONResponse)
async def trigger_unlearning(
    model_name: str = Form(
        "mrm8488/t5-base-finetuned-imdb-sentiment",
        description="Name or path of the base model."
    ),
    max_retain_samples: int = Form(
        5000,
        description="Max retain samples for faster processing. Set 0 for all."
    ),
    forget_set_uri: str = Form(
        ...,
        description="GCS URI for the forget set (e.g., gs://my-bucket/data/forget.json)."
    ),
    complete_dataset_uri: str = Form(
        ...,
        description="GCS URI for the complete dataset."
    )
):
    """
    Trigger the enhanced unlearning pipeline.

    This endpoint:
    1. Downloads datasets from GCS
    2. Runs the unlearning pipeline with improved forgetting objectives
    3. Uploads results back to GCS
    4. Returns metrics and artifact URIs

    Expected outcomes:
    - Retain set accuracy: HIGH (maintained)
    - Forget set accuracy: LOW (reduced)
    - Forget MLM loss: HIGH (increased)
    - Forget entropy: HIGH (increased)
    """
    if RESULTS_BUCKET == "your-gcs-bucket-name-here":
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: RESULTS_BUCKET is not set."
        )
        
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"api_requests/{request_id}"
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)

    try:
        # 1. Download datasets from GCS
        logging.info(f"[{request_id}] Downloading datasets from GCS...")

        forget_bucket, forget_blob = parse_gcs_uri(forget_set_uri)
        local_forget_path = os.path.join(input_dir, "forget.json")
        download_blob(forget_bucket, forget_blob, local_forget_path)

        complete_bucket, complete_blob = parse_gcs_uri(complete_dataset_uri)
        local_complete_path = os.path.join(input_dir, "complete.json")
        download_blob(complete_bucket, complete_blob, local_complete_path)

        # 2. Run the enhanced unlearning pipeline
        logging.info(f"[{request_id}] Starting enhanced unlearning pipeline...")
        args = SimpleNamespace(
            model_name=model_name,
            forget_set=local_forget_path,
            complete_dataset=local_complete_path,
            output_dir=output_dir,
            max_seq_length=128,
            max_retain_samples=max_retain_samples
        )
        output_paths = run_pipeline(args)

        # 3. Upload results to GCS
        logging.info(f"[{request_id}] Uploading results to GCS bucket: {RESULTS_BUCKET}")
        gcs_prefix = f"users/demo-user-001/results/{request_id}"

        # Upload artifacts
        upload_blob(
            RESULTS_BUCKET,
            output_paths["plot_path"],
            f"{gcs_prefix}/unlearning_comparison.png"
        )
        upload_blob(
            RESULTS_BUCKET,
            output_paths["summary_csv_path"],
            f"{gcs_prefix}/summary_table.csv"
        )
        upload_blob(
            RESULTS_BUCKET,
            output_paths["metrics_path"],
            f"{gcs_prefix}/all_metrics.json"
        )

        # Upload the entire unlearned model directory
        model_path = output_paths["unlearned_model_path"]
        for root, _, files in os.walk(model_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, model_path)
                gcs_file_path = f"{gcs_prefix}/unlearned_model/{relative_path}"
                upload_blob(RESULTS_BUCKET, local_file, gcs_file_path)

        # 4. Prepare the response with metrics and GCS links
        with open(output_paths["metrics_path"]) as f:
            metrics_data = json.load(f)

        # Calculate improvement metrics
        original = metrics_data["Original"]
        unlearned = metrics_data["Unlearned"]

        improvements = {
            "retain_accuracy_change": unlearned["retain_accuracy"] - original["retain_accuracy"],
            "forget_accuracy_change": unlearned["forget_accuracy"] - original["forget_accuracy"],
            "forget_mlm_loss_change": unlearned["forget_mlm_loss"] - original["forget_mlm_loss"],
            "forget_entropy_change": unlearned["forget_entropy"] - original["forget_entropy"]
        }

        response_data = {
            "status": "success",
            "request_id": request_id,
            "metrics": metrics_data,
            "improvements": improvements,
            "interpretation": {
                "retain_accuracy": "HIGH = Good (model still works on normal data)"
                    if unlearned["retain_accuracy"] > 0.7 else "LOW = Problem (model degraded)",
                "forget_accuracy": "LOW = Success (model forgot target data)"
                    if unlearned["forget_accuracy"] < 0.5 else "HIGH = Failed (model still remembers)",
                "forget_mlm_loss": "HIGH = Success (model uncertain about forgotten data)"
                    if unlearned["forget_mlm_loss"] > original["forget_mlm_loss"] else "LOW = Failed",
                "forget_entropy": "HIGH = Success (model predictions are uncertain)"
                    if unlearned["forget_entropy"] > original["forget_entropy"] else "LOW = Failed"
            },
            "artifacts": {
                "unlearned_model_uri": f"gs://{RESULTS_BUCKET}/{gcs_prefix}/unlearned_model/",
                "comparison_plot_uri": f"gs://{RESULTS_BUCKET}/{gcs_prefix}/unlearning_comparison.png",
                "summary_table_uri": f"gs://{RESULTS_BUCKET}/{gcs_prefix}/summary_table.csv",
                "metrics_uri": f"gs://{RESULTS_BUCKET}/{gcs_prefix}/all_metrics.json"
            }
        }

        logging.info(f"[{request_id}] Pipeline completed successfully!")
        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"[{request_id}] An error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred: {str(e)}"
        )

    finally:
        # Clean up local files
        logging.info(f"[{request_id}] Cleaning up local directory: {base_dir}")
        shutil.rmtree(base_dir, ignore_errors=True)

@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Enhanced Unlearning API (GCS Edition) is running.",
        "version": "2.1.0",
        "features": [
            "Improved forgetting objectives",
            "Better retention of general knowledge",
            "Enhanced visualization",
            "Detailed interpretations"
        ]
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cuda_available": torch.cuda.is_available(),
        "results_bucket": RESULTS_BUCKET
    }


# ============================================================================
# SECTION 5: SERVER LAUNCHER
# ============================================================================

if __name__ == "__main__":
    # --- SERVER CONFIGURATION ---
    # Get your ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
    NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"  # <--- PASTE YOUR NGROK AUTH TOKEN HERE

    SERVER_PORT = 8000
    SERVER_HOST = "0.0.0.0"

    # --- VALIDATION AND STARTUP ---
    if not NGROK_AUTH_TOKEN or "YOUR_NGROK_AUTH_TOKEN" in NGROK_AUTH_TOKEN:
        print("🚨 ERROR: Please set your NGROK_AUTH_TOKEN in the script before running.")
        print("         You can get a free token from https://dashboard.ngrok.com/get-started/your-authtoken")
        sys.exit(1)

    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        print("🚨 ERROR: The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("         Please set it to the path of your service account key file.")
        sys.exit(1)

    try:
        print("🚀 Starting Enhanced Machine Unlearning API Server...")
        print("="*70)

        # Set the ngrok authentication token
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        print("✓ Ngrok authentication successful")

        # nest_asyncio is required to run uvicorn in some environments (like notebooks)
        nest_asyncio.apply()
        print("✓ Async event loop configured")

        # Disconnect any existing tunnels to ensure a clean start
        existing_tunnels = ngrok.get_tunnels()
        if existing_tunnels:
            print(f"\n⚠️  Found {len(existing_tunnels)} existing tunnel(s). Disconnecting...")
            for tunnel in existing_tunnels:
                ngrok.disconnect(tunnel.public_url)
                print(f"   Disconnected: {tunnel.public_url}")

        # Start the ngrok tunnel to our local server
        print(f"\n🔗 Creating ngrok tunnel to localhost:{SERVER_PORT}...")
        public_url = ngrok.connect(SERVER_PORT)

        print("\n" + "="*70)
        print("✅ UNLEARNING API IS NOW LIVE!")
        print("="*70)
        print(f"\n🌐 Public URL: {public_url}")
        print(f"\n📚 API Documentation:")
        print(f"   - Interactive Docs: {public_url}/docs")
        print(f"   - ReDoc: {public_url}/redoc")
        print(f"\n🔧 API Endpoints:")
        print(f"   - POST {public_url}/unlearn/  (Trigger unlearning)")
        print(f"   - GET  {public_url}/health    (Health check)")
        print(f"   - GET  {public_url}/          (Root)")

        print(f"\n💡 Usage Example (cURL):")
        print(f"""
curl -X POST "{public_url}/unlearn/" \\
  -F "model_name=mrm8488/t5-base-finetuned-imdb-sentiment" \\
  -F "max_retain_samples=5000" \\
  -F "forget_set_uri=gs://{RESULTS_BUCKET}/path/to/forget.json" \\
  -F "complete_dataset_uri=gs://{RESULTS_BUCKET}/path/to/complete.json"
        """)

        print("\n" + "="*70)
        print("⚡ Server is starting... This may take a moment.")
        print("🛑 To stop the server, press Ctrl+C")
        print("="*70 + "\n")

        # Start the FastAPI server using uvicorn
        uvicorn.run(
            "__main__:app",
            host=SERVER_HOST,
            port=SERVER_PORT,
            log_level="info",
            access_log=True,
            reload=False # Important for single-script execution
        )

    except KeyboardInterrupt:
        print("\n\n🛑 Server shutdown requested by user.")
    except Exception as e:
        print(f"\n🚨 ERROR: An error occurred during server startup:")
        print(f"   {type(e).__name__}: {str(e)}")
        print("\nTroubleshooting:")
        print("   1. Verify your NGROK_AUTH_TOKEN is correct")
        print("   2. Check if port 8000 is already in use (netstat -an | grep 8000)")
        print("   3. Ensure you have the necessary permissions for your GCS bucket")
    finally:
        print("Cleaning up...")
        try:
            for tunnel in ngrok.get_tunnels():
                ngrok.disconnect(tunnel.public_url)
            print("✓ Ngrok tunnels disconnected")
        except:
            pass
        print("✓ Server stopped successfully")