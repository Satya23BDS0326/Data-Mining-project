"""
Fuser Module for Uncertainty-Aware Ensemble Framework (UAF)

Combines predictions from selected models into a final answer using
soft weighted voting with sharpened fusion weights.

Key improvements vs original
------------------------------
1. Soft weighted voting (was winner-take-all)
   Each model contributes f_k weight to the vote bucket of its predicted choice.
   The choice with the highest total accumulated weight wins.
   Winner-take-all fails when one high-f_k model is wrong but the majority
   of models agree on the correct answer.

2. Sharpened fusion weights
   f_k_sharp = f_k^FUSION_SHARPNESS  (default γ=2.0)
   Raw f_k = Acc_k × (1 - u_k).
   Sharpening amplifies the gap between models:
     f_k=0.8 → 0.64,  f_k=0.5 → 0.25  (gap grows from 0.30 to 0.39)
   This makes votes more decisive when one model is clearly better calibrated.
   The sharpening is monotone, so the ranking of models is preserved.
"""

import json
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import numpy as np

from config import FUSION_SHARPNESS


# ============================================================================
# Core Fusion Logic
# ============================================================================

def fuse_single_example(
    example_preds: Dict[str, Dict[str, Any]],
    acc_dict: Dict[str, float],
    sharpness: float = FUSION_SHARPNESS,
) -> Tuple[int, str, float]:
    """
    Fuse model predictions for a single example.

    Computes sharpened fusion weight for each model:
        f_k     = Acc_k × (1 - u_k)
        w_k     = f_k^sharpness

    Accumulates w_k into the vote bucket for each model's predicted choice.
    The choice with the highest total accumulated weight is selected.

    Args:
        example_preds: {model_name: {answer: int, uncertainty: float}}
        acc_dict:      {model_name: validation accuracy}
        sharpness:     exponent applied to f_k (default from config)

    Returns:
        (best_answer, best_model, best_total_weight)
            best_model = model with highest individual w_k among those that
                         voted for the winning choice (for attribution only)
    """
    vote_weights: Dict[int, float] = {}
    model_wk:     Dict[str, float] = {}

    for model_name, pred_dict in example_preds.items():
        answer      = int(pred_dict["answer"])
        uncertainty = float(pred_dict["uncertainty"])
        model_acc   = float(acc_dict.get(model_name, 0.0))

        f_k = model_acc * (1.0 - uncertainty)
        w_k = math.pow(max(f_k, 0.0), sharpness)

        model_wk[model_name] = w_k
        vote_weights[answer]  = vote_weights.get(answer, 0.0) + w_k

    best_answer = max(vote_weights, key=vote_weights.get)
    best_total  = vote_weights[best_answer]

    # Attribution: among models that voted for the winner, pick the one with
    # the highest individual weight (for model-usage statistics)
    best_model = max(
        (m for m, p in example_preds.items() if int(p["answer"]) == best_answer),
        key=lambda m: model_wk[m],
        default=next(iter(example_preds)),
    )

    return best_answer, best_model, best_total


def fuse_dataset(
    predictions: Dict[int, Dict[str, Dict[str, Any]]],
    selected_models: List[str],
    acc_dict: Dict[str, float],
    sharpness: float = FUSION_SHARPNESS,
) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Fuse predictions across the dataset using only the selected models.

    Args:
        predictions:     {example_idx: {model_name: {answer, uncertainty}}}
        selected_models: model names to include in fusion
        acc_dict:        {model_name: validation accuracy}
        sharpness:       weight exponent (default from config)

    Returns:
        (final_predictions, chosen_models)
            final_predictions: {example_idx -> predicted answer index}
            chosen_models:     {example_idx -> top-contributing model name}
    """
    final_predictions: Dict[int, int] = {}
    chosen_models:     Dict[int, str] = {}

    for example_idx, example_preds in predictions.items():
        filtered = {
            m: p for m, p in example_preds.items()
            if m in selected_models
        }
        if not filtered:
            raise ValueError(
                f"Example {example_idx}: no predictions from selected models. "
                f"Selected={selected_models}, Available={list(example_preds.keys())}"
            )

        best_answer, best_model, _ = fuse_single_example(filtered, acc_dict, sharpness)
        final_predictions[example_idx] = best_answer
        chosen_models[example_idx]     = best_model

    return final_predictions, chosen_models


# ============================================================================
# Majority Voting Baseline
# ============================================================================

def majority_vote(
    predictions: Dict[int, Dict[str, Dict[str, Any]]],
    acc_dict: Optional[Dict[str, float]] = None,
) -> Dict[int, int]:
    """
    Fuse predictions using majority voting.

    On ties, breaks by accuracy of the models that predicted each tied answer.

    Args:
        predictions: {example_idx: {model_name: {answer, uncertainty}}}
        acc_dict:    optional tie-breaker

    Returns:
        {example_idx -> predicted answer}
    """
    final_predictions: Dict[int, int] = {}

    for example_idx, example_preds in predictions.items():
        answers      = [p["answer"] for p in example_preds.values()]
        vote_counts  = Counter(answers)
        max_votes    = max(vote_counts.values())
        tied_answers = [a for a, cnt in vote_counts.items() if cnt == max_votes]

        if len(tied_answers) == 1:
            best_answer = tied_answers[0]
        elif acc_dict is not None:
            best_answer, best_acc = None, -1.0
            for ans in tied_answers:
                models_for_ans = [m for m, p in example_preds.items() if p["answer"] == ans]
                max_a = max(acc_dict.get(m, 0.0) for m in models_for_ans)
                if max_a > best_acc:
                    best_acc, best_answer = max_a, ans
        else:
            best_answer = tied_answers[0]

        final_predictions[example_idx] = best_answer

    return final_predictions


# ============================================================================
# Evaluation
# ============================================================================

def compute_accuracy(
    final_predictions: Dict[int, int],
    dataset: List[Dict[str, Any]],
) -> float:
    if not final_predictions:
        return 0.0
    correct = sum(
        1 for idx, pred in final_predictions.items()
        if dataset[idx]["mc1_targets"]["labels"][pred] == 1
    )
    return correct / len(final_predictions)


# ============================================================================
# Results Persistence
# ============================================================================

def save_predictions(final_predictions: Dict[int, int], filepath: str) -> None:
    ser = {str(idx): int(pred) for idx, pred in final_predictions.items()}
    with open(filepath, "w") as f:
        json.dump(ser, f, indent=2)
    print(f"Predictions saved to {filepath}")


def load_predictions(filepath: str) -> Dict[int, int]:
    with open(filepath) as f:
        ser = json.load(f)
    return {int(k): v for k, v in ser.items()}


def save_fusion_metadata(
    final_predictions: Dict[int, int],
    chosen_models: Dict[int, str],
    selected_models: List[str],
    accuracy: float,
    filepath: str,
) -> None:
    meta = {
        "accuracy"           : float(accuracy),
        "num_examples"       : len(final_predictions),
        "selected_models"    : selected_models,
        "num_selected_models": len(selected_models),
        "fusion_sharpness"   : FUSION_SHARPNESS,
        "per_example_choices": {str(idx): m for idx, m in chosen_models.items()},
    }
    with open(filepath, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Fusion metadata saved to {filepath}")


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_model_usage(chosen_models: Dict[int, str]) -> Dict[str, int]:
    return dict(Counter(chosen_models.values()))


def print_fusion_summary(
    final_predictions: Dict[int, int],
    dataset: List[Dict[str, Any]],
    chosen_models: Dict[int, str],
    selected_models: List[str],
) -> None:
    accuracy = compute_accuracy(final_predictions, dataset)
    usage    = analyze_model_usage(chosen_models)

    print("\n" + "="*70)
    print("FUSION RESULTS SUMMARY")
    print("="*70)
    n = len(final_predictions)
    print(f"Test Accuracy : {accuracy:.4f}  ({int(accuracy*n)}/{n})")
    print(f"Selected      : {len(selected_models)} models — {', '.join(selected_models)}")
    print(f"Sharpness γ   : {FUSION_SHARPNESS}")
    print("\nModel Usage (top contributor per example):")
    print("-"*70)
    for model_name in sorted(usage):
        cnt = usage[model_name]
        print(f"  {model_name:<40} {cnt:>6} ({100*cnt/len(chosen_models):>5.1f}%)")
    print("="*70 + "\n")
