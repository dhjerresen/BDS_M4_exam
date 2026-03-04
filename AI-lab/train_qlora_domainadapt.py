# train_qlora_domainadapt.py

# --- Imports: OS/IO + ML stack (HF Transformers, PEFT/LoRA, TRL SFT) ---
import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- Config: base model, data paths, output directory (via env vars for reproducibility) ---
MODEL_NAME  = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
DATA_PATH   = os.environ.get("DATA_PATH", "data/train_silver_domainadapt_sft.jsonl")
OUT_DIR     = os.environ.get("OUT_DIR", "outputs/qlora_patent_domainadapt")

# --- Training hyperparams: sequence length + training schedule ---
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "768"))
MAX_STEPS   = int(os.environ.get("MAX_STEPS", "800"))
WARMUP_STEPS  = int(os.environ.get("WARMUP_STEPS", "80"))
SAVE_STEPS    = int(os.environ.get("SAVE_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "20"))

# --- Optimizer/batching: small batch on GPU + gradient accumulation ---
LR       = float(os.environ.get("LR", "1e-4"))
BATCH    = int(os.environ.get("BATCH", "2"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))

# --- LoRA hyperparams: rank/alpha/dropout control adapter capacity + regularization ---
LORA_R       = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

# --- Precision toggle: fp16 vs bf16 compute (affects speed/memory/stability) ---
USE_FP16 = os.environ.get("USE_FP16", "1") == "1"

def main():
    # --- Output folder ---
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- QLoRA quantization setup: load base model in 4-bit (NF4 + double quant) ---
    compute_dtype = torch.float16 if USE_FP16 else torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4-bit weights to fit 7B model on single GPU
        bnb_4bit_quant_type="nf4",            # NF4 quantization (common choice for QLoRA)
        bnb_4bit_use_double_quant=True,       # second-stage quantization for memory savings
        bnb_4bit_compute_dtype=compute_dtype, # compute dtype for matmuls
    )

    # --- Tokenizer: chat template + pad token handling (important for packing/batching) ---
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Load base LLM in 4-bit; device_map="auto" places it on available GPU ---
    # Single GPU: device_map="auto" is OK and convenient
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    # --- Prep for k-bit training: enables stable adapter training on quantized backbone ---
    model = prepare_model_for_kbit_training(model)

    # --- LoRA injection points: target attention + MLP projection layers (Mistral-style) ---
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=LORA_R,                 # adapter rank (capacity)
        lora_alpha=LORA_ALPHA,    # scaling
        lora_dropout=LORA_DROPOUT,# regularization
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # --- Wrap model with PEFT adapters; only LoRA params become trainable ---
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Load SFT dataset: JSONL with "messages" format (chat-style supervision) ---
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # --- Convert structured chat messages -> plain text using the model's chat template ---
    def to_text(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    # --- Keep only the final "text" field for TRL SFTTrainer ---
    ds = ds.map(to_text, remove_columns=ds.column_names)

    # --- TRL SFT config: training loop settings, saving/logging, 8-bit paged AdamW ---
    args = SFTConfig(
        output_dir=OUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=USE_FP16,
        bf16=(not USE_FP16),
        optim="paged_adamw_8bit",
        report_to="none",
    )

    # --- SFTTrainer: supervised fine-tuning with packing to utilize context window efficiently ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        args=args,
        dataset_text_field="text",
        packing=True,               # packs multiple short samples into one sequence (throughput boost)
        max_seq_length=MAX_SEQ_LEN,
    )

    # --- Run training: updates only LoRA adapter weights ---
    trainer.train()

    # --- Save artifacts: LoRA adapters + tokenizer (used later for inference/agent brain) ---
    trainer.model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    # --- Save minimal metadata for reproducibility/debugging ---
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": MODEL_NAME, "data": DATA_PATH, "max_steps": MAX_STEPS}, f, indent=2)

    print(f"✅ Done. Saved QLoRA adapters to {OUT_DIR}")

# --- Script entry point ---
if __name__ == "__main__":
    main()