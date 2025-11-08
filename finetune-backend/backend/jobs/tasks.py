import os
from typing import Dict, Any
from ..trainer.train import train
from ..trainer.config import TrainConfig, LoRAConfig

from ..services.progress_service import publish

def rq_progress(run_id: str):
    def _cb(msg: Dict[str, Any]):
        publish(run_id, msg)
    return _cb

def run_training_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    run_id = payload["run_id"]
    out_dir = os.path.join(payload["artifacts_dir"], run_id)

    cfg = TrainConfig(
        run_id=run_id,
        base_model=payload["base_model"],
        load_in_4bit=payload.get("load_in_4bit", True),

        dataset=payload["dataset"],
        dataset_field_map=payload.get("dataset_field_map"),
        eval_split=payload.get("eval_split"),
        prompt_template=payload.get("prompt_template"),

        max_seq_len=payload.get("max_seq_len", 2048),
        micro_batch_size=payload.get("batch_size", 1),
        gradient_accumulation_steps=payload.get("grad_accum_steps", 4),
        epochs=payload.get("epochs", 3),
        max_steps=payload.get("max_steps", -1),
        lr=payload.get("learning_rate", 1e-4),
        weight_decay=payload.get("weight_decay", 0.01),
        warmup_steps=payload.get("warmup_steps", 5),
        lr_schedule=payload.get("lr_scheduler", "linear"),
        optimizer=payload.get("optimizer", "adamw_8bit"),
        logging_steps=payload.get("logging_steps", 10),
        seed=payload.get("seed", 3407),
        mixed_precision=payload.get("mixed_precision", "bf16"),

        lora=LoRAConfig(
            enabled=True,
            r=payload.get("lora_r", 16),
            alpha=payload.get("lora_alpha", 16),
            dropout=payload.get("lora_dropout", 0.0),
            target_modules=None,
            use_rslora=payload.get("use_rslora", False),
        ),
        output_dir=out_dir,
        save_steps=payload.get("save_steps", 200),
        save_total_limit=payload.get("save_total_limit", 3),
    )

    result = train(cfg, progress_cb=rq_progress(run_id))
    return result
