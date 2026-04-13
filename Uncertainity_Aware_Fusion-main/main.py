"""
main.py — UAF Pipeline (v3: full metric optimisation stack)

What changed vs previous version
----------------------------------
1. Four uncertainty methods instead of three
   "margin" is the gap between best and 2nd-best choice log-prob score.
   It requires no extra forward passes and is empirically the strongest
   per-example uncertainty signal for MC1 tasks.

2. Fifth method: "combined"
   After all four methods are normalised on the val set, an AUROC-weighted
   linear blend is computed per model per example.  Methods that are barely
   better than random receive near-zero weight; strong methods dominate.
   This is a free improvement: no extra inference required.

3. Augmented Haloscope features
   Each hidden-state vector is concatenated with four scalars from the model's
   choice distribution: [choice_margin_lp, choice_entropy, max_softmax, second_softmax].
   These directly encode the model's confidence and are highly predictive of
   hallucination, giving the LR probe a strong shortcut that hidden states
   alone don't have.

4. SE temperature = 0.5 (was 1.0) in BOTH val and test
   Sharpens the softmax over choice log-probs, amplifying the confident-vs-
   uncertain gap.  Consistent use keeps val normalisation valid on test.

5. Cache versioning (CACHE_VERSION = "v3")
   Cache filenames include the version prefix.  Changing any hyperparameter
   that affects cached values (SE temp, num_layers, etc.) bumps the version
   so stale caches are never silently reused.

6. Soft weighted voting with sharpened weights (see fuser.py)
   All fusion methods now use f_k^γ (γ=2.0) accumulated per choice bucket.

Run:
    python main.py
    RUN_ONLY_MODEL="gemma2:2b,phi3.5-mini" python main.py
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm

from config import (
    SEED, MODEL_LIST, K_RANGE, UNCERTAINTY_METHODS,
    CACHE_DIR, RESULTS_DIR, DEVICE, USE_MLX,
    SEMANTIC_ENTROPY_SAMPLES, SEMANTIC_ENTROPY_THRESHOLD,
    HALOSCOPE_MAX_ITER, HALOSCOPE_N_PCA, HALOSCOPE_NUM_LAYERS,
    SE_TEMPERATURE, CACHE_VERSION, VERBOSE,
)
from data_loader import load_truthfulqa, extract_mc1, format_prompt, evaluate_mc1
from model_runner import (
    load_model, unload_model,
    infer_example, sample_mc1_answers,
    predict_mc1, compute_log_probs_for_choice, get_hidden_features, generate_samples,
)
from uncertainty import PerplexityUncertainty, MarginUncertainty, SemanticEntropy, Haloscope
from selector import (
    compute_metrics, rank_models, select_top_k, tune_k,
    compute_combined_results,
    save_selector_results, print_metrics_summary,
)
from fuser import (
    fuse_dataset, majority_vote,
    save_predictions, save_fusion_metadata,
    analyze_model_usage, print_fusion_summary,
    compute_accuracy as fusion_accuracy,
)
from metrics import (
    compute_table1_auroc, compute_pairwise_matrix, compute_table3,
    plot_k_ablation, comparison_table, save_all_results,
    print_table1, print_table2, print_table3, print_comparison,
)

# ─── Setup ───────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_active_model_list() -> List[Tuple[str, str]]:
    run_only = os.getenv("RUN_ONLY_MODEL", "").strip()
    if not run_only:
        return MODEL_LIST
    requested = {n.strip() for n in run_only.split(",") if n.strip()}
    filtered  = [e for e in MODEL_LIST if e[0] in requested]
    if not filtered:
        available = ", ".join(n for n, _ in MODEL_LIST)
        raise ValueError(f"RUN_ONLY_MODEL='{run_only}' not in {available}")
    return filtered


ACTIVE_MODEL_LIST = get_active_model_list()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _choice_aug_features(choice_scores: List[float]) -> Tuple[float, float, float, float]:
    """
    Compute the four scalar features used to augment Haloscope input.

    Returns:
        (choice_margin_lp, choice_entropy, max_softmax, second_softmax)
    """
    arr = np.array(choice_scores, dtype=np.float64)
    arr_shifted = arr - arr.max()
    probs = np.exp(arr_shifted)
    probs /= probs.sum()

    sorted_probs  = np.sort(probs)[::-1]
    max_softmax   = float(sorted_probs[0])
    second_softmax = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0

    sorted_scores  = sorted(choice_scores, reverse=True)
    margin_lp      = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    entropy        = float(-np.sum(probs * np.log(probs + 1e-10)))

    return margin_lp, entropy, max_softmax, second_softmax


def _cache_path(phase: str, model_name: str) -> str:
    """Versioned cache file path.  Old versions are automatically bypassed."""
    return os.path.join(CACHE_DIR, f"{CACHE_VERSION}_{phase}_{model_name}.json")


def _is_invalid_val_cache(cached: Dict[str, Any]) -> bool:
    """
    Detect fallback-only validation caches produced when inference failed.

    Signature of invalid cache:
      - acc == 0.0
      - no hidden features
      - all perplexity raw values are null
    """
    try:
        acc = float(cached.get("acc", 0.0))
        hidden = cached.get("hidden_features")
        perp = cached.get("perplexity", {}).get("results", [])
        if not perp:
            return True

        all_null_ppl = all(e.get("raw_perplexity") is None for e in perp)
        no_hidden = not hidden
        return (acc == 0.0) and all_null_ppl and no_hidden
    except Exception:
        return True


def _is_invalid_test_cache(cached: Dict[str, Any]) -> bool:
    """
    Detect corrupt or stale test-phase caches.

    A cache is invalid when:
      - It is empty (no examples stored), OR
      - Every stored example has answer=0 across ALL methods AND at least
        10 examples exist (statistically impossible to always predict index 0
        at >2% of 735 examples if predictions were real), OR
      - The stored structure is the OLD flat format {answer, uncertainty}
        instead of the new per-method format {perplexity: {...}, ...}.

    The old flat format comes from a previous pipeline version where each
    example was stored as {"answer": N, "uncertainty": U} rather than
    {"perplexity": {"answer": N, "uncertainty": U}, ...}.  Loading the old
    format with model_preds.get(method, fallback) falls back to answer=0 for
    every example, silently producing 0% accuracy.
    """
    try:
        if not cached:
            return True

        # Check format: first entry should be a dict of method-keyed sub-dicts
        first_val = next(iter(cached.values()))
        EXPECTED_METHODS = {"perplexity", "haloscope", "semantic", "margin", "combined"}
        if not isinstance(first_val, dict):
            return True
        if not EXPECTED_METHODS.intersection(first_val.keys()):
            # Old flat format — no method keys present
            logger.warning("  Test cache appears to be old flat format; recomputing.")
            return True

        # Check if every example has answer=0 (statistically invalid for >10 examples)
        n = len(cached)
        if n > 10:
            all_zero = all(
                v.get("perplexity", {}).get("answer", -1) == 0
                for v in cached.values()
            )
            if all_zero:
                logger.warning("  Test cache has answer=0 for ALL examples; likely corrupt.")
                return True

        return False
    except Exception:
        return True


# ─── Step 1: Load Data ───────────────────────────────────────────────────────

def load_data():
    logger.info("Loading TruthfulQA ...")
    val_set, test_set = load_truthfulqa(seed=SEED)
    logger.info(f"  val={len(val_set)}  test={len(test_set)}")
    return val_set, test_set


# ─── Step 2-4: Validation Phase ──────────────────────────────────────────────

def compute_validation_results(val_set: List[dict]) -> Dict[str, Any]:
    """
    Run infer_example() once per val example per model.

    Collects per-example:
      - MC1 prediction (pred_idx)
      - Token log-probs for the predicted choice → perplexity
      - Hidden features (last HALOSCOPE_NUM_LAYERS layers, mean-pooled)
        augmented with 4 choice-distribution scalars
      - Choice scores for semantic entropy and margin
    """
    logger.info("\n" + "="*70)
    logger.info(f"VALIDATION PHASE  (SE temp={SE_TEMPERATURE}, cache={CACHE_VERSION})")

    # Initialise result dicts for all 4 inference methods
    results_per_method: Dict[str, Dict[str, List]] = {m: {} for m in UNCERTAINTY_METHODS}
    hidden_features_per_model: Dict[str, List]     = {}
    acc_dict: Dict[str, float]                     = {}

    se_scorer = SemanticEntropy(threshold=SEMANTIC_ENTROPY_THRESHOLD)

    for model_name, model_id in ACTIVE_MODEL_LIST:
        cache_file = _cache_path("val", model_name)

        if os.path.exists(cache_file):
            logger.info(f"[{model_name}] Loading val cache ({CACHE_VERSION}) ...")
            with open(cache_file) as f:
                cached = json.load(f)
            if _is_invalid_val_cache(cached):
                logger.warning(
                    f"[{model_name}] Invalid val cache detected (fallback-only). Recomputing..."
                )
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
            else:
                for method in UNCERTAINTY_METHODS:
                    if method in cached:
                        results_per_method[method][model_name] = cached[method]["results"]
                if cached.get("hidden_features"):
                    hidden_features_per_model[model_name] = cached["hidden_features"]
                acc_dict[model_name] = cached.get("acc", 0.0)
                continue

        try:
            model, tokenizer, _ = load_model(model_id, use_mlx=USE_MLX,
                                              requested_device=DEVICE)
        except Exception as e:
            logger.error(f"[{model_name}] Load failed: {e}")
            continue

        runtime_device = DEVICE

        perp_results:  List[Dict] = []
        halo_results:  List[Dict] = []
        sem_results:   List[Dict] = []
        margin_results: List[Dict] = []
        hidden_list:   List[Dict] = []
        correct_count: int = 0
        success_count: int = 0
        fail_count: int = 0

        for ex_idx, example in enumerate(tqdm(val_set, desc=model_name, unit="ex")):
            question, choices, correct_idx = extract_mc1(example)

            try:
                info = infer_example(model, tokenizer, question, choices, device=runtime_device)
                success_count += 1
            except Exception as e:
                retried = False
                if runtime_device == "mps":
                    logger.warning(
                        f"  infer_example failed on MPS ex={ex_idx}: {e}; retrying on CPU"
                    )
                    try:
                        unload_model(model)
                        model, tokenizer, _ = load_model(
                            model_id,
                            use_mlx=USE_MLX,
                            requested_device="cpu",
                        )
                        runtime_device = "cpu"
                        info = infer_example(
                            model,
                            tokenizer,
                            question,
                            choices,
                            device=runtime_device,
                        )
                        success_count += 1
                        retried = True
                    except Exception as cpu_e:
                        logger.warning(f"  CPU retry also failed ex={ex_idx}: {cpu_e}")

                if not retried:
                    fail_count += 1
                    logger.warning(f"  infer_example failed ex={ex_idx}: {e}")
                    fb = {"correct": 0, "uncertainty": 0.5}
                    perp_results.append({**fb, "raw_perplexity": None})
                    halo_results.append(fb)
                    sem_results.append({**fb, "raw_semantic_entropy": None, "choice_scores": []})
                    margin_results.append({**fb, "raw_margin": None})
                    continue

            pred_idx   = info["pred_idx"]
            is_correct = evaluate_mc1(example, pred_idx)
            correct_count += is_correct

            choice_scores = info["choice_scores"]

            # ── Perplexity ────────────────────────────────────────────────
            token_lps = info["token_log_probs"]
            raw_ppl   = float(np.exp(-np.mean(token_lps))) if token_lps else None
            perp_results.append({
                "correct"       : is_correct,
                "uncertainty"   : 0.5,          # filled after fit()
                "raw_perplexity": raw_ppl,
            })

            # ── Haloscope — augmented hidden features ─────────────────────
            hid = info["hidden_features"]
            margin_lp, entropy, max_p, second_p = _choice_aug_features(choice_scores)
            hidden_list.append({
                "idx"              : ex_idx,
                "feature"          : hid.tolist(),
                "choice_margin_lp" : margin_lp,
                "choice_entropy"   : entropy,
                "max_softmax"      : max_p,
                "second_softmax"   : second_p,
            })
            halo_results.append({"correct": is_correct, "uncertainty": 0.5})

            # ── Semantic entropy (temperature = SE_TEMPERATURE) ───────────
            try:
                raw_se = se_scorer.compute_from_choice_scores(
                    choice_scores, temperature=SE_TEMPERATURE
                )
            except Exception as exc:
                logger.warning(f"  SE failed ex={ex_idx}: {exc}")
                raw_se = None
            sem_results.append({
                "correct"              : is_correct,
                "uncertainty"          : 0.5,
                "raw_semantic_entropy" : float(raw_se) if raw_se is not None else None,
                "choice_scores"        : choice_scores,
            })

            # ── Margin ────────────────────────────────────────────────────
            margin_results.append({
                "correct"    : is_correct,
                "uncertainty": 0.5,
                "raw_margin" : margin_lp,
            })

        if success_count == 0:
            logger.error(
                f"[{model_name}] 0/{len(val_set)} successful validation inferences; "
                "excluding this model from selector/fusion."
            )
            unload_model(model)
            continue

        if fail_count:
            logger.warning(
                f"[{model_name}] Validation inference failures: {fail_count}/{len(val_set)}"
            )

        acc = correct_count / len(val_set) if val_set else 0.0
        acc_dict[model_name] = acc
        logger.info(f"[{model_name}] val accuracy = {acc:.4f}")

        results_per_method["perplexity"][model_name] = perp_results
        results_per_method["haloscope"][model_name]  = halo_results
        results_per_method["semantic"][model_name]   = sem_results
        results_per_method["margin"][model_name]     = margin_results
        if hidden_list:
            hidden_features_per_model[model_name] = hidden_list

        cache_data = {
            "acc"            : acc,
            "perplexity"     : {"results": perp_results},
            "haloscope"      : {"results": halo_results},
            "semantic"       : {"results": sem_results},
            "margin"         : {"results": margin_results},
            "hidden_features": hidden_list if hidden_list else None,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        logger.info(f"[{model_name}] Cached → {cache_file}")
        unload_model(model)

    return {
        "results_val_perplexity"   : results_per_method["perplexity"],
        "results_val_haloscope"    : results_per_method["haloscope"],
        "results_val_semantic"     : results_per_method["semantic"],
        "results_val_margin"       : results_per_method["margin"],
        "hidden_features_per_model": hidden_features_per_model,
        "acc_dict"                 : acc_dict,
    }


# ─── Step 4b: Normalize + Train Per-Model Haloscopes ─────────────────────────

def normalize_and_train(val_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    1. Fit and apply normalisation for perplexity, semantic entropy, and margin.
    2. Train one Haloscope probe per model using augmented hidden features.
    3. Compute combined uncertainty = AUROC-weighted blend of all 4 methods.
    """
    logger.info("\n" + "="*70)
    logger.info("NORMALIZING + TRAINING PER-MODEL HALOSCOPES + COMBINED METHOD")

    rv_perp   = val_data["results_val_perplexity"]
    rv_sem    = val_data["results_val_semantic"]
    rv_halo   = val_data["results_val_haloscope"]
    rv_margin = val_data["results_val_margin"]
    hidden    = val_data["hidden_features_per_model"]

    # ── 1. Perplexity normalisation ───────────────────────────────────────
    perp_scorer = PerplexityUncertainty()
    raw_ppls = [
        float(e["raw_perplexity"])
        for res in rv_perp.values()
        for e in res if e.get("raw_perplexity") is not None
    ]
    if raw_ppls:
        perp_scorer.fit(raw_ppls)
        for res in rv_perp.values():
            for e in res:
                rp = e.get("raw_perplexity")
                e["uncertainty"] = perp_scorer.normalize(float(rp)) if rp is not None else 0.5

    # ── 2. Margin normalisation ───────────────────────────────────────────
    margin_scorer = MarginUncertainty()
    raw_margins = [
        float(e["raw_margin"])
        for res in rv_margin.values()
        for e in res if e.get("raw_margin") is not None
    ]
    if raw_margins:
        margin_scorer.fit(raw_margins)
        for res in rv_margin.values():
            for e in res:
                rm = e.get("raw_margin")
                e["uncertainty"] = margin_scorer.normalize(float(rm)) if rm is not None else 0.5

    # ── 3. Haloscope — per-model probes with augmented features ──────────
    haloscopes: Dict[str, Haloscope] = {}

    for model_name, feat_items in hidden.items():
        halo_res = rv_halo.get(model_name, [])
        X_train, y_train = [], []

        for item in feat_items:
            idx  = item.get("idx")
            feat = item.get("feature")
            if idx is None or feat is None or not (0 <= idx < len(halo_res)):
                continue

            # Build augmented feature vector
            aug = Haloscope.build_augmented_features(
                hidden          = np.array(feat, dtype=np.float32),
                choice_margin_lp= float(item.get("choice_margin_lp", 0.0)),
                choice_entropy  = float(item.get("choice_entropy",   0.0)),
                max_softmax     = float(item.get("max_softmax",      0.25)),
                second_softmax  = float(item.get("second_softmax",   0.0)),
            )
            X_train.append(aug)
            y_train.append(1 - int(halo_res[idx]["correct"]))   # 1 = wrong

        if len(X_train) < 10:
            logger.warning(f"[{model_name}] Too few Haloscope samples ({len(X_train)}), skipping")
            continue

        X_arr = np.array(X_train, dtype=np.float32)
        y_arr = np.array(y_train, dtype=np.int32)
        if len(np.unique(y_arr)) < 2:
            logger.warning(f"[{model_name}] Haloscope: only one class, skipping")
            continue

        h = Haloscope(max_iter=HALOSCOPE_MAX_ITER, n_pca_components=HALOSCOPE_N_PCA)
        h.fit(X_arr, y_arr)
        haloscopes[model_name] = h

        # Backfill fitted haloscope predictions into rv_halo
        for item in feat_items:
            idx  = item.get("idx")
            feat = item.get("feature")
            if idx is None or feat is None or not (0 <= idx < len(halo_res)):
                continue
            aug = Haloscope.build_augmented_features(
                hidden          = np.array(feat, dtype=np.float32),
                choice_margin_lp= float(item.get("choice_margin_lp", 0.0)),
                choice_entropy  = float(item.get("choice_entropy",   0.0)),
                max_softmax     = float(item.get("max_softmax",      0.25)),
                second_softmax  = float(item.get("second_softmax",   0.0)),
            )
            halo_res[idx]["uncertainty"] = float(h.predict(aug))
        logger.info(f"[{model_name}] Haloscope fitted ✓")

    if not haloscopes:
        logger.warning("No per-model Haloscopes fitted — will fall back to u=0.5")

    # ── 4. Semantic entropy normalisation ─────────────────────────────────
    raw_entropies = [
        float(e["raw_semantic_entropy"])
        for res in rv_sem.values()
        for e in res if e.get("raw_semantic_entropy") is not None
    ]
    semantic_min = semantic_max = 0.0
    if raw_entropies:
        semantic_min = float(np.percentile(raw_entropies, 2))
        semantic_max = float(np.percentile(raw_entropies, 98))
        for res in rv_sem.values():
            for e in res:
                rv = e.get("raw_semantic_entropy")
                if rv is None:
                    e["uncertainty"] = 0.5
                elif semantic_max == semantic_min:
                    e["uncertainty"] = 0.0
                else:
                    e["uncertainty"] = float(
                        np.clip((float(rv) - semantic_min) / (semantic_max - semantic_min), 0, 1)
                    )

    # ── 5. Combined uncertainty (AUROC-weighted blend) ────────────────────
    logger.info("Computing combined uncertainty (AUROC-weighted blend)...")
    results_by_method_for_combined = {
        "perplexity": rv_perp,
        "haloscope" : rv_halo,
        "semantic"  : rv_sem,
        "margin"    : rv_margin,
    }
    rv_combined = compute_combined_results(results_by_method_for_combined)
    logger.info("Combined method computed ✓")

    logger.info("Normalisation + training done.")
    return {
        "results_val_perplexity" : rv_perp,
        "results_val_haloscope"  : rv_halo,
        "results_val_semantic"   : rv_sem,
        "results_val_margin"     : rv_margin,
        "results_val_combined"   : rv_combined,
        "acc_dict"               : val_data["acc_dict"],
        "perp_scorer"            : perp_scorer,
        "margin_scorer"          : margin_scorer,
        "haloscopes"             : haloscopes,
        "semantic_min"           : semantic_min,
        "semantic_max"           : semantic_max,
    }


# ─── Step 5: Model Selection ─────────────────────────────────────────────────

def run_selector(val_data: Dict[str, Any]):
    logger.info("\n" + "="*70)
    logger.info("MODEL SELECTION")

    results_by_method = {
        "perplexity": val_data["results_val_perplexity"],
        "haloscope" : val_data["results_val_haloscope"],
        "semantic"  : val_data["results_val_semantic"],
        "margin"    : val_data["results_val_margin"],
        "combined"  : val_data["results_val_combined"],
    }

    best_k_by_method:   Dict[str, int]        = {}
    selected_by_method: Dict[str, List[str]]  = {}
    k_accs_by_method:   Dict[str, Dict]       = {}
    metrics_by_method:  Dict[str, Dict]       = {}

    for method_name, results_val in results_by_method.items():
        if not results_val:
            logger.warning(f"No val results for {method_name}, skipping selector")
            continue

        method_metrics = compute_metrics(results_val)
        metrics_by_method[method_name] = method_metrics
        print_metrics_summary(method_metrics)

        max_k = min(max(K_RANGE), len(method_metrics))
        best_k, k_accs = tune_k(
            results_val,
            method_name=method_name,
            max_k=max_k,
            candidate_ks=K_RANGE,
        )

        selected = select_top_k(method_metrics, best_k)
        best_k_by_method[method_name]   = best_k
        selected_by_method[method_name] = selected
        k_accs_by_method[method_name]   = k_accs

        plot_k_ablation(k_accs,
            save_path=os.path.join(RESULTS_DIR, f"k_ablation_{method_name}.png"))
        save_selector_results(method_metrics,
            os.path.join(RESULTS_DIR, f"selector_metrics_{method_name}.csv"))
        logger.info(f"  [{method_name}] Best K={best_k}  selected={selected}")

    return best_k_by_method, selected_by_method, k_accs_by_method, metrics_by_method


# ─── Step 6: Test Phase ──────────────────────────────────────────────────────

def run_test_phase(
    test_set:      List[dict],
    perp_scorer:   PerplexityUncertainty,
    margin_scorer: MarginUncertainty,
    haloscopes:    Dict[str, Haloscope],
    semantic_min:  float,
    semantic_max:  float,
    # Combined weights per model: {model_name: {method: weight}}
    combined_weights: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Run each active model on the test set, computing all 5 uncertainty scores.
    Uses:
      - Perplexity scorer fitted on val
      - MarginUncertainty scorer fitted on val
      - Per-model Haloscope probes with augmented features
      - SE normalisation range from val (same SE_TEMPERATURE as val)
      - AUROC-weighted combined uncertainty (weights from val)
    """
    logger.info("\n" + "="*70)
    logger.info("TEST PHASE")

    se_scorer = SemanticEntropy(threshold=SEMANTIC_ENTROPY_THRESHOLD)
    ALL_METHODS = ["perplexity", "haloscope", "semantic", "margin", "combined"]
    predictions_test: Dict[str, Dict[int, Dict]] = {m: {} for m in ALL_METHODS}

    for model_name, model_id in ACTIVE_MODEL_LIST:
        cache_file = _cache_path("test", model_name)

        if os.path.exists(cache_file):
            logger.info(f"[{model_name}] Loading test cache ({CACHE_VERSION}) ...")
            with open(cache_file) as f:
                cached = json.load(f)
            if _is_invalid_test_cache(cached):
                logger.warning(
                    f"[{model_name}] Invalid test cache detected "
                    "(empty, old format, or all-zero answers). Recomputing..."
                )
                try:
                    os.remove(cache_file)   # delete so it gets rewritten after inference
                except OSError:
                    pass
            else:
                for ex_idx_str, model_preds in cached.items():
                    ex_idx = int(ex_idx_str)
                    for method in ALL_METHODS:
                        mp = model_preds.get(method, {
                            "answer": model_preds.get("answer", 0),
                            "uncertainty": 0.5,
                        })
                        predictions_test[method].setdefault(ex_idx, {})[model_name] = mp
                continue

        try:
            model, tokenizer, _ = load_model(model_id, use_mlx=USE_MLX,
                                              requested_device=DEVICE)
        except Exception as e:
            logger.error(f"[{model_name}] Load failed: {e}")
            continue

        runtime_device = DEVICE

        model_halo   = haloscopes.get(model_name)
        cw           = combined_weights.get(model_name, {})   # {method: weight}
        model_cache  = {}

        for ex_idx, example in enumerate(tqdm(test_set, desc=f"{model_name}/test", unit="ex")):
            question, choices, _ = extract_mc1(example)

            try:
                info = infer_example(model, tokenizer, question, choices, device=runtime_device)
            except Exception as e:
                retried = False
                if runtime_device == "mps":
                    logger.warning(
                        f"  infer_example failed on MPS ex={ex_idx}: {e}; retrying on CPU"
                    )
                    try:
                        unload_model(model)
                        model, tokenizer, _ = load_model(
                            model_id,
                            use_mlx=USE_MLX,
                            requested_device="cpu",
                        )
                        runtime_device = "cpu"
                        info = infer_example(
                            model,
                            tokenizer,
                            question,
                            choices,
                            device=runtime_device,
                        )
                        retried = True
                    except Exception as cpu_e:
                        logger.warning(f"  CPU retry also failed ex={ex_idx}: {cpu_e}")

                if not retried:
                    logger.warning(f"  infer_example failed ex={ex_idx}: {e}")
                    for method in ALL_METHODS:
                        predictions_test[method].setdefault(ex_idx, {})[model_name] = {
                            "answer": 0, "uncertainty": 0.5
                        }
                    continue

            pred_idx      = info["pred_idx"]
            choice_scores = info["choice_scores"]
            token_lps     = info["token_log_probs"]

            # ── Perplexity ────────────────────────────────────────────────
            raw_ppl = float(np.exp(-np.mean(token_lps))) if token_lps else None
            u_perp  = perp_scorer.normalize(raw_ppl) if (
                raw_ppl is not None and perp_scorer.min_val is not None
            ) else 0.5

            # ── Margin ────────────────────────────────────────────────────
            margin_lp, entropy, max_p, second_p = _choice_aug_features(choice_scores)
            u_margin = margin_scorer.normalize(margin_lp) if margin_scorer.min_val is not None else 0.5

            # ── Haloscope (augmented) ─────────────────────────────────────
            if model_halo is not None:
                try:
                    aug   = Haloscope.build_augmented_features(
                        hidden          = info["hidden_features"],
                        choice_margin_lp= margin_lp,
                        choice_entropy  = entropy,
                        max_softmax     = max_p,
                        second_softmax  = second_p,
                    )
                    u_halo = float(model_halo.predict(aug))
                except Exception:
                    u_halo = 0.5
            else:
                u_halo = 0.5

            # ── Semantic entropy (same SE_TEMPERATURE as val) ─────────────
            try:
                raw_se = se_scorer.compute_from_choice_scores(
                    choice_scores, temperature=SE_TEMPERATURE
                )
                if semantic_max == semantic_min:
                    u_sem = 0.0
                else:
                    u_sem = float(np.clip(
                        (raw_se - semantic_min) / (semantic_max - semantic_min), 0, 1
                    ))
            except Exception:
                u_sem = 0.5

            # ── Combined (AUROC-weighted blend of the 4 above) ────────────
            u_vals  = {"perplexity": u_perp, "haloscope": u_halo,
                       "semantic": u_sem, "margin": u_margin}
            w_sum   = sum(cw.values()) if cw else 0.0
            if w_sum > 1e-8:
                u_comb = float(sum(cw.get(m, 0.0) * u_vals[m] for m in u_vals) / w_sum)
            else:
                u_comb = float(np.mean(list(u_vals.values())))

            # ── Store in predictions_test ─────────────────────────────────
            all_u = {
                "perplexity": u_perp,
                "haloscope" : u_halo,
                "semantic"  : u_sem,
                "margin"    : u_margin,
                "combined"  : u_comb,
            }
            for method, u in all_u.items():
                predictions_test[method].setdefault(ex_idx, {})[model_name] = {
                    "answer": int(pred_idx), "uncertainty": float(u)
                }

            model_cache[ex_idx] = {m: {"answer": int(pred_idx), "uncertainty": float(u)}
                                   for m, u in all_u.items()}

        with open(cache_file, "w") as f:
            json.dump({str(k): v for k, v in model_cache.items()}, f)
        unload_model(model)

    logger.info("Test phase complete.")
    return predictions_test


# ─── Step 7: Fusion ──────────────────────────────────────────────────────────

def run_fusion(predictions_test, selected_by_method, metrics_by_method):
    logger.info("\n" + "="*70)
    logger.info("FUSION")

    fused_preds: Dict[str, Dict[int, int]] = {}

    if not metrics_by_method:
        logger.warning("No selector metrics; skipping fusion.")
        return fused_preds

    ALL_FUSION_METHODS = ["perplexity", "haloscope", "semantic", "margin", "combined"]

    for method_name in ALL_FUSION_METHODS:
        if method_name not in selected_by_method:
            continue
        if method_name not in predictions_test or not predictions_test[method_name]:
            continue

        # Use acc_dict from whatever method we have metrics for
        src_method = method_name if method_name in metrics_by_method else \
                     next(iter(metrics_by_method), None)
        if src_method is None:
            continue
        acc_dict = {m: v["acc"] for m, v in metrics_by_method[src_method].items()}

        uaf_preds, chosen = fuse_dataset(
            predictions_test[method_name],
            selected_by_method[method_name],
            acc_dict,
        )
        fused_preds[method_name] = uaf_preds
        logger.info(f"  UAF ({method_name}) usage: {analyze_model_usage(chosen)}")
        save_predictions(uaf_preds,
            os.path.join(RESULTS_DIR, f"predictions_uaf_{method_name}.json"))

    # Majority vote (on perplexity predictions as proxy for model choices)
    first_method = next(iter(metrics_by_method), "perplexity")
    baseline_acc = {m: v["acc"] for m, v in metrics_by_method[first_method].items()}
    perp_preds   = predictions_test.get("perplexity", {})
    if perp_preds:
        maj_preds = majority_vote(perp_preds, acc_dict=baseline_acc)
        fused_preds["majority"] = maj_preds
        save_predictions(maj_preds, os.path.join(RESULTS_DIR, "predictions_majority.json"))
    else:
        logger.warning("No perplexity predictions for majority voting.")

    logger.info("Fusion complete.")
    return fused_preds


# ─── K-ablation (combined method on test) ────────────────────────────────────

def compute_combined_test_k_ablation(test_set, predictions_test_combined,
                                      metrics_combined, max_k=6):
    if not metrics_combined or not predictions_test_combined:
        return {}
    ranked = [m for m, _ in sorted(metrics_combined.items(),
                                    key=lambda x: x[1]["cscore"], reverse=True)]
    acc_d  = {m: v["acc"] for m, v in metrics_combined.items()}
    k_scores = {}
    for k in range(1, min(max_k, len(ranked)) + 1):
        preds_k, _ = fuse_dataset(predictions_test_combined, ranked[:k], acc_d)
        k_scores[k] = float(fusion_accuracy(preds_k, test_set))
    return k_scores


# ─── Step 8: Metrics ─────────────────────────────────────────────────────────

def compute_all_metrics(test_set, val_data, predictions_test,
                         fused_preds, metrics_by_method, individual_preds):
    logger.info("\n" + "="*70)
    logger.info("METRICS")

    if not individual_preds:
        logger.warning("No individual predictions; skipping metrics.")
        return

    individual_accuracies = {
        model_name: sum(
            1 for idx, pred in preds.items()
            if test_set[idx]["mc1_targets"]["labels"][pred] == 1
        ) / len(preds)
        for model_name, preds in individual_preds.items()
    }

    # Table 1: AUROC — all 5 methods
    table1 = compute_table1_auroc({
        "perplexity": val_data["results_val_perplexity"],
        "haloscope" : val_data["results_val_haloscope"],
        "semantic"  : val_data["results_val_semantic"],
        "margin"    : val_data["results_val_margin"],
        "combined"  : val_data["results_val_combined"],
    })
    print_table1(table1)

    # Table 2: Pairwise (using combined uncertainties on val)
    table2 = compute_pairwise_matrix(val_data["results_val_combined"])
    print_table2(table2)

    # K-ablation on test set (combined method)
    k_scores = compute_combined_test_k_ablation(
        test_set,
        predictions_test.get("combined", {}),
        metrics_by_method.get("combined", {}),
        max_k=6,
    )
    if k_scores:
        plot_k_ablation(k_scores,
            save_path=os.path.join(RESULTS_DIR, "figure_k_ablation_combined.png"))

    # Table 3: UAF vs baselines — all 5 methods + majority
    uaf_preds_for_table3 = {
        m: fused_preds[m]
        for m in ["perplexity", "haloscope", "semantic", "margin", "combined"]
        if m in fused_preds
    }
    table3 = compute_table3(
        test_set, individual_preds,
        fused_preds.get("majority", {}),
        uaf_preds_for_table3,
    )
    print_table3(table3)

    # Comparison table (use combined metrics as primary)
    primary = "combined" if "combined" in metrics_by_method else next(iter(metrics_by_method), None)
    if primary:
        comp = comparison_table(
            metrics_by_method[primary], table3, individual_accuracies
        )
    else:
        comp = pd.DataFrame()
    print_comparison(comp)

    save_all_results(table1, table2, table3, comp, save_dir=RESULTS_DIR)
    logger.info("Metrics saved.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    logger.info("="*70)
    logger.info("UAF PIPELINE v3 — full metric optimisation stack")
    logger.info(f"Active models  : {[n for n, _ in ACTIVE_MODEL_LIST]}")
    logger.info(f"SE temperature : {SE_TEMPERATURE}")
    logger.info(f"Cache version  : {CACHE_VERSION}")
    logger.info(f"Fusion sharpness: {__import__('config').FUSION_SHARPNESS}")

    val_set, test_set = load_data()

    val_data = compute_validation_results(val_set)
    val_data = normalize_and_train(val_data)

    best_k_by_method, selected_by_method, k_accs_by_method, metrics_by_method = \
        run_selector(val_data)

    # ── Extract combined weights for test phase ───────────────────────────
    # The combined uncertainty in the test phase needs the same AUROC weights
    # used to build rv_combined.  Re-derive them from the val results.
    from sklearn.metrics import roc_auc_score as _auroc

    def _derive_combined_weights(val_data):
        """
        Re-derive the per-model AUROC weights used in compute_combined_results().
        Returns {model_name: {method: weight}} where weights sum to 1.
        """
        inference_methods = ["perplexity", "haloscope", "semantic", "margin"]
        method_results = {
            "perplexity": val_data["results_val_perplexity"],
            "haloscope" : val_data["results_val_haloscope"],
            "semantic"  : val_data["results_val_semantic"],
            "margin"    : val_data["results_val_margin"],
        }
        first_results = method_results[inference_methods[0]]
        combined_weights = {}

        for model_name in first_results:
            raw_aurocs = {}
            for m in inference_methods:
                rlist = method_results[m].get(model_name, [])
                if not rlist:
                    raw_aurocs[m] = 0.5
                    continue
                correct_vals = np.array([r["correct"]     for r in rlist])
                u_vals       = np.array([r["uncertainty"] for r in rlist])
                labels_wrong = 1 - correct_vals
                unique       = np.unique(labels_wrong)
                if len(unique) == 1:
                    raw_aurocs[m] = 0.5
                else:
                    raw_aurocs[m] = float(_auroc(labels_wrong, u_vals))

            weights = {m: max(raw_aurocs[m] - 0.5, 0.0) for m in inference_methods}
            w_sum = sum(weights.values())
            if w_sum < 1e-8:
                weights = {m: 1.0 / len(inference_methods) for m in inference_methods}
            else:
                weights = {m: v / w_sum for m, v in weights.items()}
            combined_weights[model_name] = weights

        return combined_weights

    combined_weights = _derive_combined_weights(val_data)

    predictions_test = run_test_phase(
        test_set,
        val_data["perp_scorer"],
        val_data["margin_scorer"],
        val_data["haloscopes"],
        val_data["semantic_min"],
        val_data["semantic_max"],
        combined_weights,
    )

    fused_preds = run_fusion(predictions_test, selected_by_method, metrics_by_method)

    # Build individual_preds from perplexity (pred_idx is same across methods)
    individual_preds: Dict[str, Dict[int, int]] = {}
    for model_name in dict(ACTIVE_MODEL_LIST):
        preds = {
            idx: v[model_name]["answer"]
            for idx, v in predictions_test.get("perplexity", {}).items()
            if model_name in v
        }
        if preds:
            individual_preds[model_name] = preds

    if not individual_preds:
        logger.error("No active models produced predictions.")
        logger.error("Check HuggingFace access and rerun.")
        return

    compute_all_metrics(test_set, val_data, predictions_test,
                         fused_preds, metrics_by_method, individual_preds)

    logger.info("="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
