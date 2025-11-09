import os
import json
import threading
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import HfApi
from pyngrok import ngrok
from fastapi.responses import FileResponse, JSONResponse

# === Matplotlib headless fix ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="LoRA Finetuning API (Kaggle)", version="2.0")

TRAINING_STATUS = {"running": False, "message": ""}

# === Loss Tracker Callback ===
class LossTrackerCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            self.losses.append((step, loss))

# === Finetune Request Schema ===
class FinetuneRequest(BaseModel):
    base_model: str
    dataset_url: str
    output_dir: str = "outputs"

    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 6
    weight_decay: float = 0.01
    logging_steps: int = 1
    optimizer: str = "adamw_8bit"
    lr_scheduler_type: str = "linear"

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    hf_repo_id: Optional[str] = None

# === Dataset Downloader ===
def fetch_dataset(dataset_url: str) -> str:
    """Download dataset from GCS, HTTP, or Hugging Face."""
    if dataset_url.startswith("gs://"):
        from google.cloud import storage
        import re

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/kaggle/input/jsonjson/storage-477516-c36b1095b123.json"

        match = re.match(r"gs://([^/]+)/(.+)", dataset_url)
        if not match:
            raise ValueError("Invalid GCS URL format")
        bucket_name, blob_path = match.groups()

        local_path = os.path.basename(blob_path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded dataset from GCS: {dataset_url} → {local_path}")
        return local_path

    elif dataset_url.startswith("http"):
        import requests
        local_path = os.path.basename(dataset_url)
        r = requests.get(dataset_url)
        if r.status_code != 200:
            raise Exception(f"Failed to download dataset from {dataset_url}")
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"✅ Downloaded dataset from {dataset_url}")
        return local_path

    else:
        print(f"📦 Loading Hugging Face dataset: {dataset_url}")
        try:
            dataset = load_dataset(dataset_url)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load dataset '{dataset_url}': {e}")
        local_dir = f"{dataset_url.replace('/', '_')}_dataset"
        os.makedirs(local_dir, exist_ok=True)
        dataset.save_to_disk(local_dir)
        print(f"✅ Saved dataset to: {local_dir}")
        return local_dir

# === Main Finetune Endpoint ===
@app.post("/finetune")
def finetune(request: FinetuneRequest):
    try:
        TRAINING_STATUS.update({"running": True, "message": "Training in progress..."})
        os.makedirs(request.output_dir, exist_ok=True)
        local_path = fetch_dataset(request.dataset_url)

        tokenizer = AutoTokenizer.from_pretrained(request.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("json", data_files=local_path)
        dataset = dataset["train"].train_test_split(test_size=0.1)

        model = AutoModelForCausalLM.from_pretrained(request.base_model, device_map="auto")

        # === LoRA Config ===
        peft_config = LoraConfig(
            r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=request.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

        # === Tokenization ===
        def preprocess(example):
            texts = (
                [p + " " + c for p, c in zip(example["prompt"], example["completion"])]
                if isinstance(example["prompt"], list)
                else example["prompt"] + " " + example["completion"]
            )
            return tokenizer(texts, truncation=True, padding="max_length")

        dataset = dataset.map(preprocess, batched=True)
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        # === Loss Tracker ===
        loss_tracker = LossTrackerCallback()

        training_args = TrainingArguments(
            output_dir=request.output_dir,
            gradient_accumulation_steps=request.gradient_accumulation_steps,
            num_train_epochs=request.num_train_epochs,
            warmup_steps=request.warmup_steps,
            max_steps=request.max_steps,
            logging_steps=request.logging_steps,
            weight_decay=request.weight_decay,
            learning_rate=request.learning_rate,
            lr_scheduler_type=request.lr_scheduler_type,
            optim=request.optimizer,
            fp16=True,
            report_to=[],
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[loss_tracker]
        )

        trainer.train()

        # === Plot loss curve ===
        if not loss_tracker.losses:
            raise ValueError("No losses recorded during training.")
        steps, losses = zip(*loss_tracker.losses)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, label="Training Loss", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.tight_layout()
        loss_graph_path = os.path.join(request.output_dir, "loss_curve.png")
        plt.savefig(loss_graph_path)
        plt.close()
        print(f"✅ Saved loss graph: {loss_graph_path}")

        # === Save & (optional) upload ===
        model.save_pretrained(request.output_dir)
        tokenizer.save_pretrained(request.output_dir)

        if request.hf_repo_id:
            api = HfApi()
            api.upload_folder(
                folder_path=request.output_dir,
                repo_id=request.hf_repo_id,
                repo_type="model"
            )

        TRAINING_STATUS.update({"running": False, "message": "Training completed ✅"})
        return FileResponse(loss_graph_path, media_type="image/png")

    except Exception as e:
        TRAINING_STATUS.update({"running": False, "message": f"Error: {e}"})
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/status")
def status():
    return TRAINING_STATUS

# === Start FastAPI with ngrok ===
def run_app():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    thread = threading.Thread(target=run_app, daemon=True)
    thread.start()

    ngrok.set_auth_token("YOUR_TOKEN_HERE")
    public_url = ngrok.connect(8000).public_url
    print(f"🚀 Public API URL: {public_url}")
