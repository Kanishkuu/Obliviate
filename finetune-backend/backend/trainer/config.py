from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List

LRSchedule = Literal["linear", "cosine", "constant", "cosine_with_restarts", "polynomial"]
OptimizerName = Literal["adamw_torch", "adamw_8bit", "adafactor"]
Precision = Literal["bf16", "fp16", "no"]

@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    use_rslora: bool = False

@dataclass
class TrainConfig:
    run_id: str = "run"
    base_model: str = "unsloth/Llama-3.2-1B"
    load_in_4bit: bool = True

    dataset: str = "mlabonne/FineTome-100k"
    dataset_field_map: Optional[Dict[str, str]] = None
    eval_split: Optional[str] = None
    prompt_template: Optional[str] = None

    max_seq_len: int = 2048
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_steps: int = -1
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 5
    lr_schedule: LRSchedule = "linear"
    optimizer: OptimizerName = "adamw_8bit"
    logging_steps: int = 10
    seed: int = 3407

    mixed_precision: Precision = "bf16"

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    output_dir: str = ""
    save_steps: int = 200
    save_total_limit: int = 3
