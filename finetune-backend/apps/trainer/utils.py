import os, random, numpy as np, torch, shutil
from typing import Dict, Any, Optional
from .config import LoRAConfig
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def seed_everything(seed: int):
    # Ensure all random number generators are seeded for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # Only if CUDA is actually available
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    # Although we're targeting Mac, keeping CUDA for general robustness if it somehow existed
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def prepare_model(base_model: str, load_in_4bit: bool, lora_cfg: LoRAConfig):
    device = get_device()
    # Use float16 for MPS, bfloat16 for supported CPUs/GPUs, float32 otherwise.
    # On Mac (MPS), float16 is usually the appropriate mixed precision.
    torch_dtype = torch.float16 if device == "mps" else torch.float32

    # Load model. On Mac, 4-bit loading via bitsandbytes is not available.
    # The load_in_4bit argument will be passed but effectively ignored by transformers
    # unless you integrate another 4-bit quantization library compatible with CPU/MPS.
    # For now, we'll proceed as if it's not using bitsandbytes for 4-bit.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device, # This will map to 'mps' or 'cpu'
        # Do not pass bitsandbytes-specific kwargs here as it's not installed/supported
        # unless you have a specific CPU-based 4-bit quantization method.
        # Removing load_in_4bit for now to avoid confusion as it relies on bnb
        # For Mac, we typically use Peft for LoRA, but not bnb for quantization.
    )

    if lora_cfg.enabled:
        # Determine target_modules dynamically if not provided
        if lora_cfg.target_modules is None:
            # Common target modules for attention layers in causal LMs
            # Adjust these based on the specific model architecture if needed.
            # Example for Llama:
            # query_proj, key_proj, value_proj, o_proj are common targets
            # Check model.named_modules() to find appropriate names
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            # Filter modules actually present in the model
            # This is a robust way to ensure target_modules exist
            found_target_modules = []
            for name, module in model.named_modules():
                if any(tm in name for tm in target_modules):
                    found_target_modules.append(name.split('.')[-1]) # Get the leaf module name
            lora_cfg.target_modules = list(set(found_target_modules)) # Remove duplicates

            if not lora_cfg.target_modules:
                print("WARNING: No common target modules found for LoRA. LoRA might not be applied correctly.")
                print("Consider inspecting the model architecture and manually specifying `target_modules`.")
                # Fallback to a common default if nothing found to prevent empty list issue
                lora_cfg.target_modules = ["q_proj", "v_proj"] # A safe bet for many models
            else:
                print(f"Dynamically selected LoRA target modules: {lora_cfg.target_modules}")

        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=lora_cfg.use_rslora,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Sanity check: ensure some parameters *do* require grad
        trainable_params_found = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params_found = True
                break
        if not trainable_params_found:
            raise RuntimeError("After applying LoRA, no trainable parameters were found. Check LoRA configuration and target modules.")

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
    # For MPS/CPU, merge_and_unload() might not perform full merge if model was loaded to a non-CPU device
    # but it still saves the merged weights to CPU.
    base = model.merge_and_unload()
    merged_dir = os.path.join(output_dir, "merged-model")
    base.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir