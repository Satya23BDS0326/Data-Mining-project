"""
Configuration for Uncertainty-Aware Ensemble Framework (UAF)

Tuned for Apple Silicon M3 Pro (18 GB unified memory).
Models are ordered roughly by size — smallest first so the
pipeline can be tested quickly with RUN_ONLY_MODEL.
"""

import os


def _str_to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _auto_device() -> str:
    """Choose the best available runtime device across Colab/Linux/macOS."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

# Random seed for reproducibility
SEED = 42

# ── Cache versioning ──────────────────────────────────────────────────────────
# Bump this string whenever any hyperparameter that affects cached values
# changes (SE temperature, num_layers, etc.).  Old cache files are bypassed
# automatically because their filenames won't match the new prefix.
CACHE_VERSION = os.getenv("UAF_CACHE_VERSION", "v3")

# Models to evaluate
DEFAULT_MODEL_LIST = [
    ("gemma4:e4b",   "google/gemma-4-E4B-it"),
    ("llama3.1:8b",  "meta-llama/Llama-3.1-8B-Instruct"),
    ("mistral:7b",   "mistralai/Mistral-7B-Instruct-v0.3"),
    ("qwen2.5:7b",   "Qwen/Qwen2.5-7B-Instruct"),
    ("phi3.5-mini",  "microsoft/Phi-3.5-mini-instruct"),
]

# Safer default profile for free/low-memory Colab GPUs.
COLAB_SAFE_MODEL_LIST = [
    ("phi3.5-mini",  "microsoft/Phi-3.5-mini-instruct"),
    ("gemma4:e4b",   "google/gemma-4-E4B-it"),
    ("qwen2.5:7b",   "Qwen/Qwen2.5-7B-Instruct"),
]

_model_profile = os.getenv("UAF_MODEL_PROFILE", "default").strip().lower()
MODEL_LIST = COLAB_SAFE_MODEL_LIST if _model_profile == "colab_safe" else DEFAULT_MODEL_LIST

# K-tuning range (min K, max K)
K_RANGE = [2, 3, 4]

# ── Uncertainty methods ───────────────────────────────────────────────────────
# "perplexity", "haloscope", "semantic", "margin" are the four inference-time
# methods (each computed from model outputs, no extra forward passes).
# "combined" is synthesised from the four above after val normalization.
UNCERTAINTY_METHODS = ["perplexity", "haloscope", "semantic", "margin"]

# Caching
CACHE_DIR   = os.getenv("UAF_CACHE_DIR", "cache")
RESULTS_DIR = os.getenv("UAF_RESULTS_DIR", "results")

# ── Inference settings for Apple Silicon M3 Pro ───────────────────────────────
# Device priority: UAF_DEVICE env var -> auto detect (cuda/mps/cpu).
_requested_device = os.getenv("UAF_DEVICE", "auto").strip().lower()
DEVICE  = _auto_device() if _requested_device == "auto" else _requested_device
USE_MLX = _str_to_bool(os.getenv("UAF_USE_MLX"), default=False)

# ── Semantic entropy ──────────────────────────────────────────────────────────
# temperature=0.5 sharpens the softmax over choice log-probs, amplifying the
# gap between confident (low entropy) and uncertain (high entropy) examples.
# Must be the same value in BOTH val and test phases so that semantic_min /
# semantic_max fitted on val remain valid on the test set.
SE_TEMPERATURE             = 0.5
SEMANTIC_ENTROPY_SAMPLES   = 5        # kept for API compat; unused for MC1
SEMANTIC_ENTROPY_THRESHOLD = 0.85

# ── Haloscope ─────────────────────────────────────────────────────────────────
HALOSCOPE_MAX_ITER    = 1000
# PCA reduces ~24k-dim hidden features to n_pca_components before LR.
# With only ~82 val examples, this is essential to avoid an underdetermined fit.
HALOSCOPE_N_PCA       = 64
# Number of transformer layers whose hidden states are concatenated.
# Raised from 4 → 6: richer representation, more separable for LR.
HALOSCOPE_NUM_LAYERS  = 6

# ── Fusion sharpening ─────────────────────────────────────────────────────────
# Each model's fusion weight is raised to FUSION_SHARPNESS before voting.
# f_k_sharp = (Acc_k × (1 - u_k))^FUSION_SHARPNESS
# γ=2.0: model with f_k=0.8 → weight 0.64; model with f_k=0.5 → weight 0.25.
# Amplifies gaps between models, making the soft vote more decisive.
FUSION_SHARPNESS = 2.0

# Verbosity
VERBOSE = True
