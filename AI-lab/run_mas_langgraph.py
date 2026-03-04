# run_mas_langgraph.py

# --- Imports: stdlib + typing + data + local HF inference (4-bit) + PEFT adapters ---
import json, re, time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# Config
# -----------------------
# --- Model endpoints: base + LoRA (Advocate) and separate general instruct model (Qwen) for all other roles ---
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_DIR   = "outputs/qlora_patent_domainadapt_mistral_1gpu"
QWEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# --- I/O: claims input, full JSONL output, and HITL subset CSV ---
IN_CLAIMS = "hitl_green_100_unlabeled.csv"
OUT_JSONL = "outputs/mas_labels.jsonl"
OUT_HITL  = "outputs/hitl_review.csv"

# --- HITL thresholds: control when human review is triggered ---
CONF_THRESH = 0.55
DEADLOCK_MARGIN = 1.0
DEADLOCK_MIN_MAX_STRENGTH = 7.0
DEADLOCK_MAX_CONF = 0.60
DISAGREE_HITL_MAX_CONF = 0.65
STRENGTH_VOTE_DELTA = 1.0

# --- Debug printing: show raw model outputs for first N items ---
DEBUG = True
DEBUG_N = 2

# -----------------------
# Rubrics
# -----------------------
# --- Calibration text used for (re)scoring advocate/skeptic "strength" on a consistent 0..10 scale ---
ADV_RUBRIC = (
    "adv_strength rubric:\n"
    "  0-2: no Y02/climate/mitigation/adaptation signal\n"
    "  3-5: weak/indirect signal\n"
    "  6-8: moderate signal (clear enabling tech)\n"
    "  9-10: explicit climate mitigation/adaptation purpose\n"
)
SKE_RUBRIC = (
    "ske_strength rubric:\n"
    "  0-2: claim clearly IS green (hard to argue against)\n"
    "  3-5: plausible green relevance (uncertain)\n"
    "  6-8: mostly non-green / generic tech\n"
    "  9-10: clearly non-green or greenwashing\n"
)

# -----------------------
# Prompts
# -----------------------
# --- Role prompts: schema-locked JSON outputs to make parsing deterministic ---
SYSTEM_ADV = (
    "You are the Advocate. Argue FOR labeling the claim as green (Y02-relevant).\n"
    "Return ONLY valid JSON with EXACTLY these keys:\n"
    "{\"argument\": \"...\", \"adv_strength\": 0}\n"
    "Rules:\n"
    "- argument <= 120 words\n"
    "- adv_strength integer 0..10\n"
    "- Include at least TWO short exact quotes from the claim in double quotes.\n"
    f"- {ADV_RUBRIC}\n"
)

SYSTEM_SKE = (
    "You are the Skeptic. Argue AGAINST labeling the claim as green (Y02-relevant).\n"
    "Return ONLY valid JSON with EXACTLY these keys:\n"
    "{\"argument\": \"...\", \"ske_strength\": 0}\n"
    "Rules:\n"
    "- argument <= 120 words\n"
    "- ske_strength integer 0..10\n"
    "- Include at least TWO short exact quotes from the claim in double quotes.\n"
    f"- {SKE_RUBRIC}\n"
)

SYSTEM_JUD = (
    "You are the Judge. Decide the final label based on the claim and both arguments.\n"
    "Return ONLY valid JSON with EXACT keys:\n"
    "{\"is_green\": 0 or 1, \"y02_hint\": \"\" or Y02*, \"rationale\": \"short\"}\n"
    "Rules:\n"
    "- is_green=1 ONLY if the claim clearly relates to climate mitigation/adaptation technology.\n"
    "- y02_hint MUST be \"\" unless confident it starts with \"Y02\".\n"
    "- rationale must reference at least one concrete element from the claim.\n"
)

# --- Repair system prompt: used when model outputs drift from JSON schema ---
REPAIR_SYS = (
    "You are a strict JSON reformatter.\n"
    "You ONLY output valid JSON and nothing else.\n"
    "Never include markdown. Never include explanations.\n"
    "Output must start with '{' and end with '}'."
)

# --- Scoring system prompt: forces a single integer output for rubric-based scoring ---
SCORE_SYS = "Output ONLY a single integer from 0 to 10. No other text."

# -----------------------
# Helpers
# -----------------------
# --- Utility: clamp values into a numeric range ---
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# --- Utility: enforce max length in words (used to keep arguments short & stable) ---
def trim_to_words(s: str, max_words: int = 120) -> str:
    words = (s or "").strip().split()
    return " ".join(words[:max_words])

# --- Robust JSON extraction: grab first {...} block and attempt json.loads (with simple fallback cleanup) ---
def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    cand = text[i:j+1]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        # fallback: strip trailing junk after last closing brace
        cand = re.sub(r"}\s*[^}]*$", "}", cand, flags=re.S)
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            return None

# --- Parse helper: coerce to int {0,1} else default ---
def as_int_0_1(x, default=None):
    try:
        v = int(x)
        return v if v in (0, 1) else default
    except Exception:
        return default

# --- Parse helper: coerce strengths to float in [0,10] ---
def as_strength_0_10(x, default=5.0) -> float:
    try:
        return clamp(float(x), 0.0, 10.0)
    except Exception:
        return default

# --- Quality gate: require at least 2 quoted spans -> >=4 quote characters ---
def has_two_quotes(arg: str) -> bool:
    return (arg or "").count('"') >= 4

# --- Extract an integer score 0..10 from model output (for scoring/repair paths) ---
def extract_first_int_0_10(text: str) -> Optional[int]:
    m = re.search(r"(-?\d+)", text or "")
    if not m:
        return None
    try:
        return int(clamp(float(int(m.group(1))), 0.0, 10.0))
    except Exception:
        return None

# --- Convert advocate vs skeptic margin into a confidence proxy (simple heuristic mapping) ---
def strength_to_confidence(adv: float, ske: float) -> float:
    margin = abs(float(adv) - float(ske))
    conf = 0.55 + 0.04 * margin
    return clamp(conf, 0.40, 0.95)

# --- Keyword heuristics: used only to detect polarity mismatches in strengths vs argument text ---
GREEN_CUES = [
    "emission", "co2", "carbon", "renewable", "solar", "wind", "hydrogen",
    "energy efficiency", "efficiency", "capture", "sequestration",
    "mitigation", "adaptation", "low-carbon", "decarbon", "climate",
    "battery", "grid", "heat pump", "insulation"
]
NONGREEN_CUES = ["not green", "non-green", "no climate", "generic", "unrelated", "greenwashing"]

def looks_green(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in GREEN_CUES)

def looks_non_green(text: str) -> bool:
    t = (text or "").lower()
    return any(c in t for c in NONGREEN_CUES)

# --- Y02 sanitization: allow only plausible Y02 prefixes and strip anything else ---
Y02_PREFIXES = ("Y02A", "Y02B", "Y02C", "Y02D", "Y02E", "Y02P", "Y02T", "Y02W")

def sanitize_y02_hint(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    m = re.search(r"\b(Y02[A-Z][0-9A-Z/.\-]*)\b", s)
    if not m:
        return ""
    tok = m.group(1)
    return tok if any(tok.startswith(p) for p in Y02_PREFIXES) else ""

# -----------------------
# Models
# -----------------------
class LocalChatModel:
    """Generic local HF chat model in 4-bit."""
    # --- Loads a general-purpose instruct model (Qwen) in 4-bit for fast local inference ---
    def __init__(self, model_name: str):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

    @torch.no_grad()
    def chat(self, system: str, user: str, *, max_new_tokens=220) -> str:
        # --- Standard chat-format prompting using the tokenizer's chat template ---
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        input_len = enc["input_ids"].shape[1]

        # --- Deterministic decoding (do_sample=False) for repeatable labeling ---
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        completion_ids = out[0][input_len:]
        return self.tok.decode(completion_ids, skip_special_tokens=True).strip()

class LocalQLoRAAdvocate:
    """QLoRA model used ONLY for Advocate role."""
    # --- Loads base model in 4-bit + attaches LoRA adapters trained on patent domain ---
    def __init__(self, base_model: str, lora_dir: str):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, lora_dir).eval()

    @torch.no_grad()
    def chat(self, system: str, user: str, *, max_new_tokens=240) -> str:
        # --- Same chat wrapper as LocalChatModel, but running the LoRA-augmented model ---
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        input_len = enc["input_ids"].shape[1]

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        completion_ids = out[0][input_len:]
        return self.tok.decode(completion_ids, skip_special_tokens=True).strip()

# -----------------------
# Qwen helpers: scoring + repair
# -----------------------
# --- Use Qwen as a "grader" to convert an argument + rubric into a stable integer strength ---
def qwen_score(qwen: LocalChatModel, rubric: str, claim: str, argument: str, kind: str) -> int:
    raw = qwen.chat(
        SCORE_SYS,
        f"Score {kind} (0..10).\n{rubric}\nReturn ONLY one integer 0..10.\n\nCLAIM:\n{claim}\n\nARGUMENT:\n{argument}\n",
        max_new_tokens=8,
    )
    v = extract_first_int_0_10(raw)
    return int(v if v is not None else 5)

# --- Repair functions: if schema drifts, force Qwen to rewrite into exact JSON schema ---
def repair_adv(qwen: LocalChatModel, claim: str, raw: str) -> Dict[str, Any]:
    prompt = (
        "Rewrite into EXACT JSON with keys {\"argument\":\"...\",\"adv_strength\":0-10}.\n"
        "HARD RULES:\n"
        "- MUST include BOTH keys\n"
        "- adv_strength MUST be integer 0..10\n"
        "- argument MUST be <= 120 words\n"
        "- Include at least TWO quotes from the CLAIM (double quotes)\n"
        "- Output ONLY JSON\n\n"
        f"CLAIM:\n{claim}\n\nMODEL_OUTPUT:\n{raw}\n"
    )
    last = qwen.chat(REPAIR_SYS, prompt, max_new_tokens=240)
    obj = extract_json_obj(last) or {}
    obj.setdefault("argument", "")
    obj.setdefault("adv_strength", 5)
    obj["argument"] = trim_to_words(str(obj.get("argument") or ""), 120)
    return obj

def repair_ske(qwen: LocalChatModel, claim: str, raw: str) -> Dict[str, Any]:
    prompt = (
        "Rewrite into EXACT JSON with keys {\"argument\":\"...\",\"ske_strength\":0-10}.\n"
        "HARD RULES:\n"
        "- MUST include BOTH keys\n"
        "- ske_strength MUST be integer 0..10\n"
        "- argument MUST be <= 120 words\n"
        "- Include at least TWO quotes from the CLAIM (double quotes)\n"
        "- Output ONLY JSON\n\n"
        f"CLAIM:\n{claim}\n\nMODEL_OUTPUT:\n{raw}\n"
    )
    last = qwen.chat(REPAIR_SYS, prompt, max_new_tokens=240)
    obj = extract_json_obj(last) or {}
    obj.setdefault("argument", "")
    obj.setdefault("ske_strength", 5)
    obj["argument"] = trim_to_words(str(obj.get("argument") or ""), 120)
    return obj

def repair_judge(qwen: LocalChatModel, claim: str, adv: Dict[str, Any], ske: Dict[str, Any], raw: str) -> Dict[str, Any]:
    prompt = (
        "Rewrite into EXACT JSON with keys {\"is_green\":0 or 1,\"y02_hint\":\"\" or Y02*,\"rationale\":\"short\"}.\n"
        "HARD RULES:\n"
        "- MUST include all keys\n"
        "- y02_hint MUST be \"\" unless confident it starts with \"Y02\"\n"
        "- rationale must reference at least one concrete element from the CLAIM\n"
        "- Output ONLY JSON\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"ADVOCATE:\n{adv}\n\nSKEPTIC:\n{ske}\n\nMODEL_OUTPUT:\n{raw}\n"
    )
    last = qwen.chat(REPAIR_SYS, prompt, max_new_tokens=220)
    obj = extract_json_obj(last) or {}
    obj.setdefault("is_green", None)
    obj.setdefault("y02_hint", "")
    obj.setdefault("rationale", "")
    return obj

# -----------------------
# MAS steps
# -----------------------
def run_adv(adv_model: LocalQLoRAAdvocate, qwen: LocalChatModel, claim: str) -> Tuple[Dict[str, Any], str]:
    # --- Advocate step: run domain-adapted QLoRA model with schema-lock user prompt ---
    # ✅ schema-lock in USER prompt (important for QLoRA drift)
    user = f"""
You MUST output ONLY a JSON object with EXACT keys:
{{"argument":"...", "adv_strength":0}}

If you output ANY other keys (e.g., is_green, y02_hint, rationale), you FAIL.

CLAIM:
{claim}
"""
    raw0 = adv_model.chat(SYSTEM_ADV, user, max_new_tokens=240)
    obj = extract_json_obj(raw0)
    if not (isinstance(obj, dict) and "argument" in obj and "adv_strength" in obj):
        # --- Repair path: use Qwen to rewrite into exact schema if parsing fails ---
        obj = repair_adv(qwen, claim, raw0)

    # --- Normalize fields: cap length and clamp strength range ---
    arg = trim_to_words(str(obj.get("argument") or ""), 120)
    adv_strength = as_strength_0_10(obj.get("adv_strength"), default=5.0)

    # --- Enforce evidence: require >=2 exact quotes from claim, else repair once more ---
    # quote enforcement -> if no >=2 quotes, repair again once
    if not has_two_quotes(arg):
        obj = repair_adv(qwen, claim, json.dumps(obj, ensure_ascii=False))
        arg = trim_to_words(str(obj.get("argument") or ""), 120)
        adv_strength = as_strength_0_10(obj.get("adv_strength"), default=adv_strength)

    # --- Polarity-aware rescoring: if strength seems inconsistent with text cues, rescore using Qwen rubric ---
    # polarity-aware rescoring
    if (adv_strength <= 1.0 and len(arg) > 20) or (adv_strength <= 3.0 and looks_green(arg)):
        adv_strength = float(qwen_score(qwen, ADV_RUBRIC, claim, arg, "adv_strength"))

    adv = {"argument": arg, "adv_strength": float(adv_strength)}
    return adv, raw0

def run_ske(qwen: LocalChatModel, claim: str) -> Tuple[Dict[str, Any], str]:
    # --- Skeptic step: Qwen plays skeptic (generally better instruction following than local QLoRA) ---
    # Skeptic is Qwen itself (more role-following)
    user = f"""
You MUST output ONLY a JSON object with EXACT keys:
{{"argument":"...", "ske_strength":0}}

If you output ANY other keys, you FAIL.

CLAIM:
{claim}
"""
    raw0 = qwen.chat(SYSTEM_SKE, user, max_new_tokens=240)
    obj = extract_json_obj(raw0)
    if not (isinstance(obj, dict) and "argument" in obj and "ske_strength" in obj):
        # --- Repair path: enforce schema ---
        obj = repair_ske(qwen, claim, raw0)

    # --- Normalize fields: length and numeric bounds ---
    arg = trim_to_words(str(obj.get("argument") or ""), 120)
    ske_strength = as_strength_0_10(obj.get("ske_strength"), default=5.0)

    # --- Evidence requirement: at least two direct quotes from claim ---
    if not has_two_quotes(arg):
        obj = repair_ske(qwen, claim, json.dumps(obj, ensure_ascii=False))
        arg = trim_to_words(str(obj.get("argument") or ""), 120)
        ske_strength = as_strength_0_10(obj.get("ske_strength"), default=ske_strength)

    # --- Polarity-aware rescoring: fix cases where text argues non-green but score is too low ---
    if (ske_strength <= 1.0 and len(arg) > 20) or (ske_strength <= 3.0 and looks_non_green(arg)):
        ske_strength = float(qwen_score(qwen, SKE_RUBRIC, claim, arg, "ske_strength"))

    ske = {"argument": arg, "ske_strength": float(ske_strength)}
    return ske, raw0

def run_judge(qwen: LocalChatModel, claim: str, adv: Dict[str, Any], ske: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    # --- Judge step: Qwen decides final label; also compute a strength-based "vote" for disagreement checks ---
    adv_strength = as_strength_0_10(adv.get("adv_strength"), 5.0)
    ske_strength = as_strength_0_10(ske.get("ske_strength"), 5.0)
    strength_pred = 1 if (adv_strength - ske_strength) >= STRENGTH_VOTE_DELTA else 0

    # --- Provide full context: claim + both arguments + their strengths ---
    user = (
        "Decide the final label.\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"ADVOCATE (strength={adv_strength}):\n{adv.get('argument','')}\n\n"
        f"SKEPTIC (strength={ske_strength}):\n{ske.get('argument','')}\n\n"
        "Return ONLY JSON."
    )
    raw0 = qwen.chat(SYSTEM_JUD, user, max_new_tokens=200)
    obj = extract_json_obj(raw0)
    if not (isinstance(obj, dict) and "is_green" in obj and "y02_hint" in obj and "rationale" in obj):
        # --- Repair path: enforce schema if judge drifts ---
        obj = repair_judge(qwen, claim, adv, ske, raw0)

    # --- Postprocess outputs: coerce 0/1, sanitize Y02 hint, keep rationale ---
    is_green = as_int_0_1(obj.get("is_green"), default=None)
    y02_hint = sanitize_y02_hint(str(obj.get("y02_hint") or ""))
    rationale = str(obj.get("rationale") or "").strip()

    # ✅ judge fallback: never None
    # --- Fail-safe: if judge returns invalid label, fall back to strength vote to avoid missing labels ---
    if is_green is None:
        is_green = int(strength_pred)
        if not rationale:
            rationale = "Fallback: judge JSON/format failure; used strength comparison."

    jud = {"is_green": int(is_green), "y02_hint": y02_hint, "rationale": rationale}
    return jud, raw0

# -----------------------
# HITL decision
# -----------------------
def needs_hitl(adv: Dict[str, Any], ske: Dict[str, Any], jud: Dict[str, Any]) -> Tuple[bool, bool, float, int, bool]:
    # --- Compute confidence proxy and detect deadlock/disagreement conditions ---
    adv_strength = as_strength_0_10(adv.get("adv_strength"), 5.0)
    ske_strength = as_strength_0_10(ske.get("ske_strength"), 5.0)
    conf = strength_to_confidence(adv_strength, ske_strength)

    # --- Secondary prediction from strengths: used to detect judge-vs-strength disagreement ---
    strength_pred = 1 if (adv_strength - ske_strength) >= STRENGTH_VOTE_DELTA else 0
    judge_pred = int(jud.get("is_green", 0))
    disagree = (judge_pred != strength_pred)

    # --- Deadlock: strengths are close AND both are high-ish AND confidence still low ---
    deadlock = (
        abs(adv_strength - ske_strength) <= DEADLOCK_MARGIN
        and max(adv_strength, ske_strength) >= DEADLOCK_MIN_MAX_STRENGTH
        and conf <= DEADLOCK_MAX_CONF
    )

    # --- Trigger HITL: either deadlock or low confidence; plus disagreement gating at low-ish confidence ---
    needs = deadlock or (conf < CONF_THRESH)
    if disagree and conf < DISAGREE_HITL_MAX_CONF:
        needs = True

    return bool(needs), bool(deadlock), float(conf), int(strength_pred), bool(disagree)

# -----------------------
# Main
# -----------------------
def main():
    # --- Ensure outputs folder exists ---
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # --- Load claim CSV (expects 'text' column; uses doc_id if present) ---
    df = pd.read_csv(IN_CLAIMS)
    assert "text" in df.columns, "CSV must contain a 'text' column."
    id_col = "doc_id" if "doc_id" in df.columns else None
    n = len(df)
    print(f"Loaded {n} claims from {IN_CLAIMS}", flush=True)

    # --- Load models once (expensive); reuse for all claims ---
    print("Loading QLoRA Advocate (base + LoRA)...", flush=True)
    t0 = time.time()
    qlora_adv = LocalQLoRAAdvocate(BASE_MODEL, LORA_DIR)
    print(f"QLoRA Advocate loaded in {time.time()-t0:.1f}s", flush=True)

    print("Loading Qwen (Skeptic/Judge/Repair/Scoring)...", flush=True)
    t1 = time.time()
    qwen = LocalChatModel(QWEN_MODEL)
    print(f"Qwen loaded in {time.time()-t1:.1f}s", flush=True)

    # --- Collect only the cases that need human review into a separate CSV ---
    hitl_rows = []
    dbg_left = DEBUG_N

    # --- Stream results to JSONL (one record per claim) for easy downstream parsing ---
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            claim_id = str(row[id_col]) if id_col else str(i)
            claim = str(row["text"])

            print(f"[{i+1:03d}/{n}] id={claim_id} running...", flush=True)
            t_claim = time.time()

            # --- Run the MAS: Advocate -> Skeptic -> Judge ---
            adv, adv_raw0 = run_adv(qlora_adv, qwen, claim)
            ske, ske_raw0 = run_ske(qwen, claim)
            jud, jud_raw0 = run_judge(qwen, claim, adv, ske)

            # --- Decide whether this item is routed to HITL based on confidence/deadlock/disagreement ---
            needs, deadlock, conf, strength_pred, disagree = needs_hitl(adv, ske, jud)

            # --- Persist full record (structured outputs + minimal raw for debugging) ---
            record = {
                "claim_id": claim_id,
                "text": claim,
                "advocate": adv,
                "skeptic": ske,
                "judge": jud,
                "confidence": conf,
                "deadlock": deadlock,
                "needs_human_review": needs,
                "strength_pred": strength_pred,
                "strength_judge_disagree": disagree,
                # minimal raw for debugging
                "advocate_raw_initial": adv_raw0,
                "skeptic_raw_initial": ske_raw0,
                "judge_raw_initial": jud_raw0,
            }

            # --- Optional debug print for first DEBUG_N claims ---
            if DEBUG and dbg_left > 0:
                dbg_left -= 1
                print("\n--- DEBUG ---", flush=True)
                print("ADV_RAW:", adv_raw0[:400], flush=True)
                print("SKE_RAW:", ske_raw0[:400], flush=True)
                print("JUD_RAW:", jud_raw0[:400], flush=True)
                print(
                    f"adv={adv.get('adv_strength')} ske={ske.get('ske_strength')} "
                    f"strength_pred={strength_pred} judge={jud.get('is_green')} "
                    f"conf={conf:.2f} deadlock={deadlock} disagree={disagree} HITL={needs}\n",
                    flush=True
                )

            # --- Write JSONL line immediately (safer if run crashes mid-way) ---
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{i+1:03d}/{n}] id={claim_id} done in {time.time()-t_claim:.1f}s | "
                f"is_green={jud.get('is_green')} conf={conf:.2f} deadlock={deadlock} disagree={disagree} HITL={needs}",
                flush=True
            )

            # --- If HITL needed: append a row with all context + empty human label fields ---
            if needs:
                hitl_rows.append({
                    "claim_id": claim_id,
                    "text": claim,
                    "advocate_argument": adv.get("argument", ""),
                    "adv_strength": adv.get("adv_strength", ""),
                    "skeptic_argument": ske.get("argument", ""),
                    "ske_strength": ske.get("ske_strength", ""),
                    "judge_is_green": jud.get("is_green"),
                    "strength_pred": strength_pred,
                    "strength_judge_disagree": disagree,
                    "confidence": conf,
                    "deadlock": deadlock,
                    "judge_y02_hint": jud.get("y02_hint", ""),
                    "judge_rationale": jud.get("rationale", ""),
                    "human_is_green_gold": "",
                    "human_note": "",
                })

    # --- Export HITL subset for manual review in spreadsheet/CSV workflow ---
    pd.DataFrame(hitl_rows).to_csv(OUT_HITL, index=False)
    print(f"✅ Wrote MAS labels: {OUT_JSONL}", flush=True)
    print(f"✅ Wrote HITL review CSV: {OUT_HITL} (n={len(hitl_rows)})", flush=True)

# --- Script entry point ---
if __name__ == "__main__":
    main()
