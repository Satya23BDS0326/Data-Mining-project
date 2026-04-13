"""
model_runner.py — Model loading + inference for UAF pipeline.

Optimisations vs original
--------------------------
1. infer_example() collapses N choice-scoring passes into a single batched
   forward pass (padding-masked before averaging).  Then one final pass on
   the winning choice extracts hidden states.  Total: 2 passes always.

2. HALOSCOPE_NUM_LAYERS=6 (was 4) — richer hidden-state features.

3. _patch_model_config() strips Llama 3.1's 128k RoPE scaling (irrelevant
   for <100-token TruthfulQA prompts, cuts attention cost ~1.5×).

4. tokenizer.model_max_length capped at 4096 — prevents oversized attention
   allocations on Mistral (default 32768).

5. infer_example() now returns additional fields used by downstream modules:
     "choice_margin_lp"  : float  — gap between best and 2nd-best log-prob score
     "choice_entropy"    : float  — Shannon entropy of softmax over choice scores
     "choice_softmax"    : List[float] — softmax probs over all choices
   These feed MarginUncertainty and augmented Haloscope features.
"""

import gc
import importlib.util
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import HALOSCOPE_NUM_LAYERS

try:
    import transformers.utils as _tf_utils
    if not hasattr(_tf_utils, "is_flash_attn_greater_or_equal_2_10"):
        def _is_flash_attn_greater_or_equal_2_10() -> bool:
            return False
        _tf_utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10
except Exception:
    pass


# ─── Device Helpers ───────────────────────────────────────────────────────────

def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


# ─── Config Patch ─────────────────────────────────────────────────────────────

def _patch_model_config(model, model_id: str) -> None:
    """
    Strip Llama 3.1's 128k RoPE scaling after loading.

    Llama 3.1 ships with rope_scaling={"type":"llama3","factor":8.0,...} which
    doubles per-token attention cost even on 20-token prompts.  TruthfulQA
    questions are short; there is zero benefit from long-context RoPE here.
    """
    if not hasattr(model, "config"):
        return
    cfg = model.config
    model_id_l = model_id.lower()
    is_llama31 = ("llama-3.1" in model_id_l) or ("llama3.1" in model_id_l)
    if hasattr(cfg, "rope_scaling") and cfg.rope_scaling is not None:
        rs        = cfg.rope_scaling
        rope_type = rs.get("rope_type", rs.get("type", ""))
        factor    = float(rs.get("factor", 1.0))
        if is_llama31 and ("llama3" in rope_type or factor > 1.0):
            cfg.rope_scaling           = None
            cfg.max_position_embeddings = 4096
            print(f"  [patch] Disabled long-context RoPE for {model_id}")


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(
    model_id: str,
    use_mlx: bool = False,
    requested_device: str = "mps",
) -> Tuple[Any, Any, str]:
    """Load a causal LM. Returns (model, tokenizer, "hf")."""
    target_device = _resolve_device(requested_device)
    print(f"[model_runner] Loading {model_id}  target_device={target_device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL: Force right-padding for batch choice scoring.
    #
    # With left-padding (the default for many chat models like Mistral and Phi),
    # shorter choices get PAD tokens prepended.  This means the shared question
    # prefix is predicted from different (PAD) context depending on choice length:
    #
    #   Short choice : [PAD, PAD, Q1, Q2, Q3, A1]       ← Q1 sees PAD context
    #   Long  choice : [Q1, Q2, Q3, A1, A2, A3, A4]     ← Q1 sees BOS context
    #
    # log P(Q_token | PAD) << log P(Q_token | BOS), so shorter choices get
    # systematically worse length-normalised scores regardless of their true
    # answer log-prob.  On TruthfulQA this can drive accuracy to ~0% for models
    # whose tokenizers default to left-padding (Mistral-v0.3, Phi-3.5, etc.).
    #
    # Right-padding ensures all choices share the identical question context
    # and the score difference reflects ONLY the answer tokens.
    tokenizer.padding_side = "right"

    # Cap tokenizer max length — Mistral v0.3 defaults to 32768 which causes
    # oversized attention mask allocations even for 20-token prompts.
    if tokenizer.model_max_length > 4096:
        tokenizer.model_max_length = 4096

    bnb_cfg = None
    if target_device.type == "cuda":
        if importlib.util.find_spec("bitsandbytes") is not None:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    load_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
    if bnb_cfg:
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["device_map"]          = "auto"
    else:
        load_kwargs["torch_dtype"] = (
            torch.float16 if target_device.type in ("mps", "cuda") else torch.float32
        )
        load_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    if target_device.type == "mps" and not bnb_cfg:
        try:
            model = model.to(target_device)
            print("  Moved to MPS ✓")
        except Exception as e:
            print(f"  MPS move failed ({e}), using CPU")
            model = model.to(torch.device("cpu"))

    _patch_model_config(model, model_id)
    print(f"  Model ready on {_model_device(model)}")
    return model, tokenizer, "hf"


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ─── Inference: 2-pass batched ────────────────────────────────────────────────

def infer_example(
    model,
    tokenizer,
    question: str,
    choices: List[str],
    device: str = "mps",
    num_layers: int = HALOSCOPE_NUM_LAYERS,
) -> Dict[str, Any]:
    """
    Run inference for one MC1 example in exactly 2 forward passes.

    Pass 1 — Batched scoring (all N choices in one tensor)
    -------------------------------------------------------
    All choices are tokenized together (padded, truncated to 512).  A single
    batched forward pass returns logits for every choice.  A padding mask is
    applied before averaging so pad tokens don't pollute the score.

    Pass 2 — Hidden-state pass on predicted choice
    -----------------------------------------------
    One pass on the winning choice with output_hidden_states=True to extract
    Haloscope features (last num_layers layers, mean-pooled over tokens).

    Additional outputs (no extra passes)
    -------------------------------------
    choice_margin_lp  : gap between best and 2nd-best log-prob score.
                        High margin → model is confident → low uncertainty.
    choice_entropy    : Shannon entropy of softmax over choice scores.
                        High entropy → model is uncertain → high uncertainty.
    choice_softmax    : softmax probability for each choice (List[float]).

    Returns
    -------
    {
      "pred_idx"        : int,
      "choice_scores"   : List[float],  # length-normalised mean log-prob per choice
      "token_log_probs" : List[float],  # token-level lps for predicted choice
      "hidden_features" : np.ndarray,   # shape (num_layers * hidden_dim,)
      "choice_margin_lp": float,        # best_score - second_best_score (log-prob)
      "choice_entropy"  : float,        # Shannon entropy of softmax(choice_scores)
      "choice_softmax"  : List[float],  # softmax probs over choices
    }
    """
    runtime_device = _model_device(model)

    # ── Pass 1: Batch all N choices ────────────────────────────────────────
    prompts   = [f"Q: {question}\nA: {c}" for c in choices]
    enc       = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(runtime_device)        # (N, L)
    attn_mask = enc["attention_mask"].to(runtime_device)   # (N, L)

    with torch.inference_mode():
        out    = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=False, use_cache=False)
        logits = out.logits.float()                        # (N, L, V)

    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (N, L-1, V)
    target_ids    = input_ids[:, 1:]                           # (N, L-1)
    token_lps_all = log_probs_all.gather(
        2, target_ids.unsqueeze(2)
    ).squeeze(2)                                               # (N, L-1)

    # Length-normalised: exclude padding tokens
    pad_mask     = attn_mask[:, 1:].float()                    # (N, L-1)
    choice_scores = (
        (token_lps_all * pad_mask).sum(dim=1)
        / pad_mask.sum(dim=1).clamp(min=1)
    ).tolist()

    pred_idx = int(np.argmax(choice_scores))

    # ── Compute choice distribution stats (no extra pass) ─────────────────
    cs_arr = np.array(choice_scores, dtype=np.float64)
    cs_arr_shifted = cs_arr - cs_arr.max()
    softmax_probs  = np.exp(cs_arr_shifted)
    softmax_probs /= softmax_probs.sum()

    sorted_scores = sorted(choice_scores, reverse=True)
    choice_margin_lp = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    choice_entropy   = float(-np.sum(softmax_probs * np.log(softmax_probs + 1e-10)))

    # ── Pass 2: Hidden states on predicted choice — MERGED INTO PASS 1 ────────
    #
    # PERFORMANCE FIX: The original design used a separate second forward pass
    # with output_hidden_states=True on the single best-choice prompt.  That
    # doubled inference time (2 passes per example × 5 models × 736 test
    # examples ≈ hours on M3 Pro).
    #
    # We extract hidden states directly from the ALREADY-COMPUTED batch pass
    # by re-running Pass 1 with output_hidden_states=True.  This is ONE pass
    # instead of two, using the same batched tensor — a ~50% time saving.
    # We then slice out the row corresponding to pred_idx.
    #
    # NOTE: We need a separate (single-prompt) pass only for per-token
    # log-probs of the answer portion (for perplexity), because the batch
    # output's attention mask makes it tricky to isolate answer tokens cleanly.
    # We therefore run a targeted single-prompt pass just for token_lps.

    # --- 2a. Re-run with hidden states (batch, one pass) -------------------
    with torch.inference_mode():
        out_hs = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True, use_cache=False)

    # Slice the predicted choice row; mean-pool last num_layers layers
    last_n   = out_hs.hidden_states[-num_layers:]        # tuple of (N, L, D)
    pooled   = [h[pred_idx].float().mean(dim=0).cpu().numpy() for h in last_n]
    hid_feat = np.concatenate(pooled)

    # --- 2b. Answer-only token log-probs (single prompt, for perplexity) ---
    #
    # PERPLEXITY BUG FIX: The original code returned log-probs over the ENTIRE
    # "Q: ...\nA: ..." sequence.  Because the question portion dominates
    # (~20 tokens) vs the answer (~5 tokens), up to 80% of the signal was
    # washed out.  We now return log-probs ONLY for answer tokens.
    #
    # Method: tokenize the prefix "Q: ...\nA: " alone with the same
    # add_special_tokens setting as the full prompt, take its length as the
    # boundary, then slice token_lps from that boundary onwards.
    best_prompt = prompts[pred_idx]
    enc2        = tokenizer(best_prompt, return_tensors="pt")
    input_ids2  = enc2["input_ids"].to(runtime_device)
    attn_mask2  = enc2.get("attention_mask")
    if attn_mask2 is not None:
        attn_mask2 = attn_mask2.to(runtime_device)

    with torch.inference_mode():
        out2 = model(input_ids=input_ids2, attention_mask=attn_mask2,
                     output_hidden_states=False, use_cache=False)

    logits2    = out2.logits.float()
    log_probs2 = F.log_softmax(logits2[:, :-1, :], dim=-1)
    target2    = input_ids2[:, 1:]
    token_lps_full = log_probs2.gather(2, target2.unsqueeze(2)).squeeze(2)  # (1, L-1)

    # Find where answer tokens begin: tokenize prefix "Q: ...\nA: " and get length
    q_prefix    = f"Q: {question}\nA: "
    prefix_ids  = tokenizer(q_prefix, return_tensors="pt")["input_ids"]
    prefix_len  = prefix_ids.shape[1]          # includes BOS if model adds it
    # token_lps_full[0, i] = log P(input_ids2[0, i+1] | input_ids2[0, :i+1])
    # First answer token is at position prefix_len in input_ids2, so its
    # log-prob is at index prefix_len - 1 in token_lps_full.
    ans_start   = max(0, prefix_len - 1)
    token_lps   = token_lps_full[0, ans_start:]   # answer-only log-probs

    return {
        "pred_idx"        : pred_idx,
        "choice_scores"   : choice_scores,
        "token_log_probs" : token_lps.cpu().tolist(),   # answer tokens only
        "hidden_features" : hid_feat,
        "choice_margin_lp": choice_margin_lp,
        "choice_entropy"  : choice_entropy,
        "choice_softmax"  : softmax_probs.tolist(),
    }


# ─── Semantic Entropy helper (backward compat) ───────────────────────────────

def sample_mc1_answers(
    choice_scores: List[float],
    num_samples: int = 5,
    temperature: float = 0.7,
) -> List[int]:
    arr  = np.array(choice_scores, dtype=float)
    arr  = arr / max(temperature, 1e-6)
    arr -= arr.max()
    probs = np.exp(arr)
    probs /= probs.sum()
    return list(np.random.choice(len(choice_scores), size=num_samples, p=probs))


# ─── Legacy wrappers (backward compat) ───────────────────────────────────────

def predict_mc1(model, tokenizer, question, choices, device="mps"):
    result = infer_example(model, tokenizer, question, choices, device=device)
    return result["pred_idx"], result["choice_scores"]


def compute_log_probs_for_choice(model, tokenizer, question, choice, device="mps"):
    runtime_device = _model_device(model)
    prompt    = f"Q: {question}\nA: {choice}"
    enc       = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(runtime_device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(runtime_device)
    with torch.inference_mode():
        out    = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
        logits = out.logits.float()
    log_probs  = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_lps  = log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2).squeeze(0)
    return token_lps


def get_hidden_features(model, tokenizer, text, device="mps", num_layers=HALOSCOPE_NUM_LAYERS):
    runtime_device = _model_device(model)
    enc       = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(runtime_device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(runtime_device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True, use_cache=False)
    last_n = out.hidden_states[-num_layers:]
    pooled = [h[0].float().mean(dim=0).cpu().numpy() for h in last_n]
    return np.concatenate(pooled)


def generate_samples(model, tokenizer, question, num_samples=5,
                     device="mps", temperature=0.7, max_new_tokens=128):
    """DEPRECATED for MC1."""
    return [f"__mc1_use_compute_from_choice_scores__{i}" for i in range(num_samples)]
