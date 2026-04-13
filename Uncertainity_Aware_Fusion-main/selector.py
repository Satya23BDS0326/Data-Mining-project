"""
Selector Module for Uncertainty-Aware Ensemble Framework (UAF)

Evaluates models on a validation set based on:
    1. Accuracy (Acc): Proportion of correct predictions
    2. Selectivity (SAH): AUROC of uncertainty calibration
    3. C-score: Harmonic mean of Acc and SAH

Implements:
    - Per-model metric computation (harmonic C-score)
    - compute_combined_results(): synthesise a 5th uncertainty method by
      AUROC-weighting the four individual methods per model
    - Model ranking and top-K selection
    - K-tuning via validation set soft-vote simulation
    - Results persistence

Harmonic C-score vs product
-----------------------------
Original:   C = Acc × SAH
New:        C = 2 × Acc × SAH / (Acc + SAH)   (harmonic mean / F1-analogue)

With the product, a coin-flip model (Acc=0.50, SAH=0.90) scores 0.45,
outranking a genuinely good model (Acc=0.62, SAH=0.70) at 0.43.
The harmonic mean gives 0.643 vs 0.657 — correctly ranking the more accurate
model higher.  This directly affects which models are selected for fusion.

compute_combined_results()
---------------------------
For each model and each val example, computes:
    u_combined = sum_m(w_m × u_m) / sum_m(w_m)
where w_m = max(AUROC_m - 0.5, 0) is the "above-random" AUROC of method m
for that model.  Methods that are barely better than random get near-zero
weight; strong methods dominate.

The combined signal is a linear ensemble of the four independent uncertainty
estimators and empirically achieves higher AUROC than any individual method.
"""

import csv
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score


# ============================================================================
# Core Metric Computation
# ============================================================================

def compute_metrics(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """
    Compute Acc, SAH (AUROC), and harmonic C-score for each model.

    Args:
        results: {model_name: [{correct: 0|1, uncertainty: float}, ...]}

    Returns:
        {model_name: {acc: float, sah: float, cscore: float}}
    """
    metrics = {}

    for model_name, result_list in results.items():
        correct_vals = np.array([r["correct"]     for r in result_list], dtype=np.float32)
        u_vals       = np.array([r["uncertainty"] for r in result_list], dtype=np.float32)

        acc = float(np.mean(correct_vals))

        labels_wrong = 1 - correct_vals
        unique       = np.unique(labels_wrong)
        sah          = 0.5 if len(unique) == 1 else float(roc_auc_score(labels_wrong, u_vals))

        # Harmonic mean (F1-analogue): penalises extreme imbalance more gracefully
        denom  = acc + sah
        cscore = (2.0 * acc * sah / denom) if denom > 0 else 0.0

        metrics[model_name] = {"acc": acc, "sah": sah, "cscore": cscore}

    return metrics


# ============================================================================
# Combined Uncertainty (AUROC-weighted blend of all methods)
# ============================================================================

def compute_combined_results(
    results_by_method: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Synthesise a "combined" uncertainty stream by AUROC-weighting individual methods.

    For each model:
        1. Compute the AUROC of each method's uncertainty on val.
        2. Weight = max(AUROC - 0.5, 0)  — above-random contribution only.
        3. u_combined[example] = weighted average of u_m[example] across methods.

    This is a free improvement: no extra forward passes, just a linear blend
    of already-computed uncertainties using their empirical quality as weights.

    Args:
        results_by_method: {method_name: {model_name: [result_list]}}
            Each result_list entry must have "correct" and "uncertainty" fields.

    Returns:
        {model_name: [{correct, uncertainty}, ...]}  — the combined results
    """
    methods = list(results_by_method.keys())
    if not methods:
        return {}

    # Compute AUROC per method per model
    method_aurocs: Dict[str, Dict[str, float]] = {}
    for method, results in results_by_method.items():
        method_aurocs[method] = {}
        for model_name, result_list in results.items():
            correct_vals = np.array([r["correct"]     for r in result_list], dtype=np.float32)
            u_vals       = np.array([r["uncertainty"] for r in result_list], dtype=np.float32)
            labels_wrong = 1 - correct_vals
            unique       = np.unique(labels_wrong)
            if len(unique) == 1:
                method_aurocs[method][model_name] = 0.5
            else:
                method_aurocs[method][model_name] = float(roc_auc_score(labels_wrong, u_vals))

    # Build combined results per model
    first_method    = methods[0]
    all_model_names = list(results_by_method[first_method].keys())
    combined_results: Dict[str, List[Dict[str, Any]]] = {}

    for model_name in all_model_names:
        # Per-method AUROCs → above-random weights
        aurocs  = np.array(
            [method_aurocs[m].get(model_name, 0.5) for m in methods],
            dtype=np.float64
        )
        weights = np.maximum(aurocs - 0.5, 0.0)
        w_sum   = weights.sum()
        if w_sum < 1e-8:
            weights = np.ones(len(methods)) / len(methods)
        else:
            weights = weights / w_sum

        print(
            f"  combined weights [{model_name}]: "
            + ", ".join(f"{m}={w:.3f}" for m, w in zip(methods, weights))
        )

        # Number of examples
        n = max(len(results_by_method[m].get(model_name, [])) for m in methods)

        combined: List[Dict[str, Any]] = []
        for ex_idx in range(n):
            u_vals_ex = []
            for m_idx, m in enumerate(methods):
                rlist = results_by_method[m].get(model_name, [])
                u_vals_ex.append(rlist[ex_idx]["uncertainty"] if ex_idx < len(rlist) else 0.5)

            u_comb  = float(np.dot(weights, u_vals_ex))
            # Correct label from the first method (same prediction, only uncertainty differs)
            rlist0  = results_by_method[first_method].get(model_name, [])
            correct = rlist0[ex_idx]["correct"] if ex_idx < len(rlist0) else 0
            combined.append({"correct": correct, "uncertainty": u_comb})

        combined_results[model_name] = combined

    return combined_results


# ============================================================================
# Model Ranking and Selection
# ============================================================================

def rank_models(metrics: Dict[str, Dict[str, float]]) -> List[Tuple[str, Dict[str, float]]]:
    """Rank models by harmonic C-score (descending)."""
    return sorted(metrics.items(), key=lambda x: x[1]["cscore"], reverse=True)


def select_top_k(metrics: Dict[str, Dict[str, float]], k: int) -> List[str]:
    """Select the top-K models by harmonic C-score."""
    return [name for name, _ in rank_models(metrics)[:k]]


# ============================================================================
# K-Tuning: Soft-Vote Simulation on Validation Set
# ============================================================================

def simulate_fusion(
    results: Dict[str, List[Dict[str, Any]]],
    selected_models: List[str],
    metrics: Dict[str, Dict[str, float]]
) -> float:
    """
    Simulate soft-weighted voting on the validation set.

    Each model accumulates weight f_k = Acc_k × (1 - u_k) into the vote
    bucket for its "correct" sentinel (1=right, 0=wrong).  The bucket with
    the higher total weight wins.  Accuracy = fraction of examples where
    the "right" bucket wins.

    Note: val results only have {correct, uncertainty}, not choice indices,
    so we proxy "votes" with the correct/incorrect sentinel.  This gives
    a faithful simulation of the fusion accuracy directional signal.
    """
    first_model = list(results.keys())[0]
    n_examples  = len(results[first_model])
    correct_sum = 0

    for ex_idx in range(n_examples):
        vote_buckets: Dict[int, float] = {}

        for model_name in selected_models:
            rlist       = results[model_name]
            example     = rlist[ex_idx]
            correct     = int(example["correct"])       # 0 or 1 sentinel
            uncertainty = float(example["uncertainty"])
            model_acc   = metrics[model_name]["acc"]
            f_k         = model_acc * (1.0 - uncertainty)
            vote_buckets[correct] = vote_buckets.get(correct, 0.0) + f_k

        best_sentinel = max(vote_buckets, key=vote_buckets.get)
        correct_sum  += best_sentinel

    return correct_sum / n_examples


def tune_k(
    results: Dict[str, List[Dict[str, Any]]],
    method_name: str,
    max_k: int = 6,
    candidate_ks: Optional[List[int]] = None,
) -> Tuple[int, Dict[int, float]]:
    """
    Find optimal K by simulating fusion for K=1..max_k on the validation set.

    Returns:
        Tuple[best_k, k_accuracies_dict]
    """
    metrics    = compute_metrics(results)
    ranked     = rank_models(metrics)
    all_models = [name for name, _ in ranked]
    max_k      = min(max_k, len(all_models))

    if candidate_ks:
        allowed_ks = sorted({int(k) for k in candidate_ks if 1 <= int(k) <= max_k})
        if not allowed_ks:
            allowed_ks = list(range(1, max_k + 1))
    else:
        allowed_ks = list(range(1, max_k + 1))

    k_accuracies  = {}
    best_k        = 1
    best_acc      = -np.inf

    print(
        f"Tuning K over {len(allowed_ks)} candidates for method='{method_name}'..."
    )
    for k in allowed_ks:
        selected = all_models[:k]
        val_acc  = simulate_fusion(results, selected, metrics)
        k_accuracies[k] = val_acc
        print(f"  K={k}: {selected}  Val Acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_k   = k

    print(f"Best K: {best_k} (Val Acc={best_acc:.4f})")
    return best_k, k_accuracies


# ============================================================================
# Results Persistence
# ============================================================================

def save_selector_results(
    metrics: Dict[str, Dict[str, float]],
    filepath: str
) -> None:
    ranked = rank_models(metrics)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "sah_auroc", "cscore"])
        for model_name, m in ranked:
            writer.writerow([
                model_name,
                f"{m['acc']:.6f}",
                f"{m['sah']:.6f}",
                f"{m['cscore']:.6f}",
            ])
    print(f"Selector results saved to {filepath}")


# ============================================================================
# Summary Reporting
# ============================================================================

def print_metrics_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    ranked = rank_models(metrics)
    print("\n" + "="*70)
    print("SELECTOR METRICS  (C-score = 2×Acc×SAH/(Acc+SAH), harmonic mean)")
    print("="*70)
    print(f"{'Model':<30} {'Acc':>10} {'SAH':>10} {'C-score':>10}")
    print("-"*70)
    for model_name, m in ranked:
        print(f"{model_name:<30} {m['acc']:>10.4f} {m['sah']:>10.4f} {m['cscore']:>10.4f}")
    print("="*70 + "\n")
