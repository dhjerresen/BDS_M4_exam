# train_qlora_domainadapt.py

# --- Imports: standard libs + PyTorch + HF datasets/transformers + PEFT(LoRA) + TRL(SFT) ---
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Config: base model, data, output (override via env vars for reproducibility) ---
MODEL_NAME  = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
DATA_PATH   = os.environ.get("DATA_PATH", "data/train_silver_domainadapt_sft.jsonl")
OUT_DIR     = os.environ.get("OUT_DIR", "outputs/qlora_patent_domainadapt")

# --- Sequence length used by SFT (packing + truncation) ---
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "768"))

# --- Training schedule: either fixed max_steps or epoch-based training ---
MAX_STEPS     = int(os.environ.get("MAX_STEPS", "-1"))   # if >0, overrides epochs
EPOCHS        = float(os.environ.get("EPOCHS", "1"))

# --- Warmup: allow either explicit warmup_steps OR warmup_ratio (mutually exclusive in config below) ---
WARMUP_STEPS  = int(os.environ.get("WARMUP_STEPS", "0"))
WARMUP_RATIO  = float(os.environ.get("WARMUP_RATIO", "0.03"))

# --- Logging/checkpointing cadence ---
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "20"))

# --- LoRA hyperparams: adapter capacity + regularization ---
LORA_R       = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# --- Optimizer/batching: small microbatch + gradient accumulation to reach larger effective batch ---
LR       = float(os.environ.get("LR", "1e-4"))
BATCH    = int(os.environ.get("BATCH", "2"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))

def main():
    # --- Ensure output directory exists ---
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- QLoRA quantization config: load base model in 4-bit NF4, compute in bf16 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- Tokenizer: right padding and pad token fallback (needed for batching/packing) ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load quantized base model ---
    # ✅ DDP-safe: DO NOT use device_map="auto" (DDP expects each process controls its own device)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    # --- Prepare model for k-bit training: stability tweaks for adapter training on quantized weights ---
    model = prepare_model_for_kbit_training(model)

    # --- Where LoRA is applied: attention projections + MLP projections (Mistral architecture) ---
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # --- LoRA config: only these injected adapter weights become trainable (base stays frozen) ---
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # --- Wrap model with PEFT; print how many params are trainable (sanity check) ---
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Load SFT dataset: JSONL with chat "messages" structure ---
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # --- Convert chat messages -> single training string using model chat template ---
    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    # --- Keep only "text" for TRL trainer input ---
    ds = ds.map(to_text, remove_columns=ds.column_names)

    # --- Choose between step-based training vs epoch-based training ---
    use_max_steps = MAX_STEPS > 0

    # --- TRL SFTConfig: schedule, warmup policy, saving/logging, bf16, and 8-bit paged AdamW ---
    sft_args = SFTConfig(
        output_dir=OUT_DIR,
        max_steps=(MAX_STEPS if use_max_steps else -1),
        num_train_epochs=(EPOCHS if not use_max_steps else 1),

        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",

        # --- Warmup: if warmup_steps is set, ratio is forced to 0 (avoid double-warmup) ---
        warmup_steps=(WARMUP_STEPS if WARMUP_STEPS > 0 else 0),
        warmup_ratio=(0.0 if WARMUP_STEPS > 0 else WARMUP_RATIO),

        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # --- SFTTrainer: supervised fine-tuning; packing packs multiple samples per sequence for throughput ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_args,
        dataset_text_field="text",
        packing=True,
        max_seq_length=MAX_SEQ_LEN,
    )

    # --- Train adapters (LoRA weights) on patent-style instruction/chat data ---
    trainer.train()

    # --- Save adapters + tokenizer for later inference (e.g., as agent "brain") ---
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    # --- Log run metadata for reproducibility / report (hyperparams + data + schedule) ---
    meta = {
        "base_model": MODEL_NAME,
        "data": DATA_PATH,
        "max_seq_len": MAX_SEQ_LEN,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lr": LR,
        "epochs": EPOCHS,
        "batch": BATCH,
        "grad_acc": GRAD_ACC,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "save_steps": SAVE_STEPS,
        "logging_steps": LOGGING_STEPS,
    }

    # --- Persist meta JSON next to checkpoints ---
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Done. Saved QLoRA adapters to {OUT_DIR}")

# --- Script entry point ---
if __name__ == "__main__":
    main()
