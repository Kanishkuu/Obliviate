from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

LRSchedule = Literal["linear", "cosine", "constant", "cosine_with_restarts", "polynomial"]
OptimizerName = Literal["adamw_torch", "adamw_8bit", "adafactor"]

class FineTuneRequest(BaseModel):
    # Identifiers
    run_id: Optional[str] = None

    # Data
    dataset: str = Field(..., description="HF dataset name or local path to .jsonl/.json/.csv")
    dataset_field_map: Optional[Dict[str, str]] = Field(
        default=None, description='e.g. {"prompt": "input", "completion": "output"}'
    )
    eval_split: Optional[str] = None
    prompt_template: Optional[str] = None  # e.g. "### Instruction: {instruction}\n### Response:"

    # Model
    base_model: str = "unsloth/Llama-3.2-1B"
    load_in_4bit: bool = True

    # Training params
    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 4
    epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_scheduler: LRSchedule = "linear"
    optimizer: OptimizerName = "adamw_8bit"
    logging_steps: int = 10
    seed: int = 3407
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False

    # Saving
    save_steps: int = 200
    save_total_limit: int = 3

class JobResponse(BaseModel):
    run_id: str
    job_id: str

class ResultResponse(BaseModel):
    run_id: str
    result: Dict[str, Any]
