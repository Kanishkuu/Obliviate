from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class FineTuneRequest(BaseModel):
    run_id: Optional[str] = None
    dataset: str
    base_model: str
    load_in_4bit: bool = True

    max_seq_len: int = 2048
    batch_size: int = 1
    grad_accum_steps: int = 4
    epochs: int = 3
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_scheduler: str = "linear"
    optimizer: str = "adamw_8bit"
    logging_steps: int = 10
    seed: int = 3407
    mixed_precision: str = "bf16"

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_rslora: bool = False

    save_steps: int = 200
    save_total_limit: int = 3
    eval_split: Optional[str] = None
    dataset_field_map: Optional[Dict[str, str]] = None
    prompt_template: Optional[str] = None

class JobResponse(BaseModel):
    run_id: str
    job_id: str

class ResultResponse(BaseModel):
    run_id: str
    result: Dict[str, Any]
