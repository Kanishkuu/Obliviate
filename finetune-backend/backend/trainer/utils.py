import os
import random, numpy as np, torch
from typing import Dict, Any, Optional
from .config import LoRAConfig
from peft import LoraConfig as PeftLoRAConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_model(base_model: str, load_in_4bit: bool, lora_cfg: LoRAConfig):
    kwargs = {}
    if load_in_4bit:
        kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ))

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": "cpu"},
        **kwargs
    )

    if lora_cfg.enabled:
        peft_config = PeftLoRAConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=lora_cfg.use_rslora,
        )
        model = get_peft_model(model, peft_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass
    return model

def save_artifacts(trainer, output_dir: str) -> Dict[str, Any]:
    paths = {}
    adapter_dir = os.path.join(output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    trainer.tokenizer.save_pretrained(adapter_dir)
    paths["adapter_dir"] = adapter_dir
    return paths

def maybe_merge_lora(model, tokenizer, output_dir: str, merge: bool) -> Optional[str]:
    if not isinstance(model, PeftModel) or not merge:
        return None
    base = model.merge_and_unload()
    merged_dir = os.path.join(output_dir, "merged-model")
    base.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir
