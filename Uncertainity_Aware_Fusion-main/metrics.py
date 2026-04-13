"""
Evaluation Metrics Module for Uncertainty-Aware Ensemble Framework (UAF)

Computes all evaluation tables and plots for research paper:
    - Table 1: AUROC per uncertainty method (now includes margin + combined)
    - Table 2: Pairwise model variation matrix
    - Table 3: UAF vs baselines (test set accuracy, all methods)
    - K-ablation plot
    - Comparison table (before/after UAF)

Updated display_names in compute_table3() to handle the new "margin" and
"combined" fusion methods in addition to the original three.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from sklearn.metrics import roc_auc_score


# ============================================================================
# Table 1: AUROC per Uncertainty Method
# ============================================================================

def compute_table1_auroc(
    results_per_method: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> pd.DataFrame:
    """
    Compute AUROC for each model under each uncertainty method.

    Args:
        results_per_method: {method_name: {model_name: [result_list]}}
            Each result has "correct" and "uncertainty" fields.

    Returns:
        DataFrame with columns [model, <method1>, <method2>, ...]
        Values are AUROC scores rounded to 2 decimal places.
    """
    if not results_per_method:
        return pd.DataFrame()

    first_method = next(iter(results_per_method))
    model_names  = sorted(results_per_method[first_method].keys())
    rows         = []

    for model_name in model_names:
        row = {"model": model_name}
        for method_name, results_by_model in results_per_method.items():
            if model_name not in results_by_model:
                row[method_name] = np.nan
                continue
            result_list   = results_by_model[model_name]
            correct_vals  = np.array([r["correct"]     for r in result_list], dtype=np.float32)
            u_vals        = np.array([r["uncertainty"] for r in result_list], dtype=np.float32)
            labels_wrong  = 1 - correct_vals
            unique        = np.unique(labels_wrong)
            auroc         = 0.5 if len(unique) == 1 else roc_auc_score(labels_wrong, u_vals)
            row[method_name] = round(float(auroc), 2)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Preferred column order — any subset of these that exist
    preferred = ["model", "perplexity", "haloscope", "semantic", "margin", "combined"]
    cols      = [c for c in preferred if c in df.columns]
    extra     = [c for c in df.columns if c not in cols]
    df        = df[cols + extra]
    return df


# ============================================================================
# Table 2: Pairwise Model Variation Matrix
# ============================================================================

def compute_pairwise_matrix(
    results_val: Dict[str, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Compute pairwise comparison matrix between models.

    Cell (i, j) = "(a, b)" where:
        a = % examples where model-j is correct and model-i is wrong
        b = % examples where model-j detects a hallucination model-i misses
            (detect = wrong AND uncertainty > 0.5)

    Returns:
        NxN DataFrame; diagonal cells are "NA".
    """
    model_names = sorted(results_val.keys())
    n           = len(model_names)
    if n == 0:
        return pd.DataFrame()

    first_model = model_names[0]
    n_examples  = len(results_val[first_model])
    if n_examples == 0:
        return pd.DataFrame(index=model_names, columns=model_names)

    matrix = np.empty((n, n), dtype=object)

    for i, mi in enumerate(model_names):
        res_i = results_val[mi]
        for j, mj in enumerate(model_names):
            if i == j:
                matrix[i, j] = "NA"
                continue
            res_j        = results_val[mj]
            acc_comp     = 0
            det_comp     = 0
            for ex in range(n_examples):
                ci = res_i[ex]["correct"]
                ui = float(res_i[ex]["uncertainty"])
                cj = res_j[ex]["correct"]
                uj = float(res_j[ex]["uncertainty"])
                if cj == 1 and ci == 0:
                    acc_comp += 1
                if (cj == 0 and uj > 0.5) and (ci == 0 and ui <= 0.5):
                    det_comp += 1
            pct_a = round(100.0 * acc_comp / n_examples, 1)
            pct_d = round(100.0 * det_comp / n_examples, 1)
            matrix[i, j] = f"({pct_a:.1f}, {pct_d:.1f})"

    return pd.DataFrame(matrix, index=model_names, columns=model_names)


# ============================================================================
# Table 3: UAF vs Baselines
# ============================================================================

def compute_table3(
    test_dataset: List[Dict[str, Any]],
    individual_model_preds: Dict[str, Dict[int, int]],
    majority_preds: Dict[int, int],
    uaf_preds_dict: Dict[str, Dict[int, int]],
) -> pd.DataFrame:
    """
    Compare accuracy across ensemble strategies on the test set.

    Args:
        test_dataset:           Ground-truth examples
        individual_model_preds: {model_name: {example_idx: predicted_answer}}
        majority_preds:         {example_idx: predicted_answer}
        uaf_preds_dict:         {method_name: {example_idx: predicted_answer}}
            Supported method names: perplexity, haloscope, semantic, margin, combined

    Returns:
        DataFrame with columns [method, accuracy (%)]
    """
    results = []

    # Best individual model
    indiv_accs = {m: compute_accuracy(p, test_dataset) for m, p in individual_model_preds.items()}
    if indiv_accs:
        best_model = max(indiv_accs, key=indiv_accs.get)
        results.append({
            "method"  : f"Best Individual ({best_model})",
            "accuracy": round(100.0 * indiv_accs[best_model], 2),
        })

    # Majority voting
    if majority_preds:
        maj_acc = compute_accuracy(majority_preds, test_dataset)
        results.append({"method": "Majority Voting", "accuracy": round(100.0 * maj_acc, 2)})

    # UAF variants
    display_names = {
        "uaf"       : "UAF",
        "perplexity": "UAF (Perplexity)",
        "haloscope" : "UAF (Haloscope)",
        "semantic"  : "UAF (Semantic)",
        "margin"    : "UAF (Margin)",        # NEW
        "combined"  : "UAF (Combined)",      # NEW
    }
    for method_name, preds in uaf_preds_dict.items():
        if not preds:
            continue
        acc = compute_accuracy(preds, test_dataset)
        results.append({
            "method"  : display_names.get(method_name, f"UAF ({method_name})"),
            "accuracy": round(100.0 * acc, 2),
        })

    return pd.DataFrame(results)


# ============================================================================
# Helper: Accuracy
# ============================================================================

def compute_accuracy(
    predictions: Dict[int, int],
    dataset: List[Dict[str, Any]],
) -> float:
    if not predictions:
        return 0.0
    correct = sum(
        1 for idx, pred in predictions.items()
        if dataset[idx]["mc1_targets"]["labels"][pred] == 1
    )
    return correct / len(predictions)


# ============================================================================
# K-Ablation Plot
# ============================================================================

def plot_k_ablation(
    k_scores: Dict[int, float],
    save_path: str = "results/k_ablation.png",
) -> None:
    if not k_scores:
        return
    ks   = sorted(k_scores.keys())
    accs = [k_scores[k] for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, accs, marker="o", linestyle="-", linewidth=2, markersize=8)
    plt.xlabel("Number of Models (K)", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title("K-Ablation: Accuracy vs Ensemble Size", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xticks(ks)
    if max(accs) > min(accs):
        plt.ylim([min(accs) * 0.95, max(accs) * 1.02])

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"K-ablation plot saved to {save_path}")
    plt.close()


# ============================================================================
# Comparison Table: Before / After UAF
# ============================================================================

def comparison_table(
    selector_metrics: Dict[str, Dict[str, float]],
    table3_results: pd.DataFrame,
    individual_accuracies: Dict[str, float],
) -> pd.DataFrame:
    """
    Create comparison table: individual models vs UAF ensemble.

    Returns DataFrame with columns:
        Method | Accuracy (%) | AUROC | C-score | Delta vs Best Individual (%)
    """
    rows = []
    best_model = max(individual_accuracies, key=individual_accuracies.get)
    best_acc   = individual_accuracies[best_model]

    for model_name, metrics in selector_metrics.items():
        acc_pct = round(individual_accuracies.get(model_name, 0.0) * 100, 2)
        rows.append({
            "Method"              : model_name,
            "Accuracy (%)"        : acc_pct,
            "AUROC"               : round(metrics["sah"], 3),
            "C-score"             : round(metrics["cscore"], 3),
            "Delta vs Best (%)"   : round(acc_pct - best_acc * 100, 2),
        })

    for _, row in table3_results.iterrows():
        acc_pct = row["accuracy"]
        rows.append({
            "Method"            : row["method"],
            "Accuracy (%)"      : acc_pct,
            "AUROC"             : "-",
            "C-score"           : "-",
            "Delta vs Best (%)": round(acc_pct - best_acc * 100, 2),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Save All Results
# ============================================================================

def save_all_results(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    comparison: pd.DataFrame,
    save_dir: str = "results",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for df, fname in [
        (table1,      "table1_auroc.csv"),
        (table3,      "table3_uaf_vs_baselines.csv"),
        (comparison,  "comparison_before_after_uaf.csv"),
    ]:
        path = os.path.join(save_dir, fname)
        df.to_csv(path, index=False)
        print(f"Saved {path}")
    path2 = os.path.join(save_dir, "table2_pairwise.csv")
    table2.to_csv(path2, index=True)
    print(f"Saved {path2}")


# ============================================================================
# Print Formatted Tables
# ============================================================================

def print_table1(table1: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 1: AUROC per Uncertainty Method")
    print("="*80)
    print(table1.to_string(index=False))
    print("="*80 + "\n")


def print_table2(table2: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 2: Pairwise Model Variation (%)")
    print("="*80)
    print(table2.to_string())
    print("="*80 + "\n")


def print_table3(table3: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("TABLE 3: UAF vs Baselines (Test Set Accuracy %)")
    print("="*80)
    print(table3.to_string(index=False))
    print("="*80 + "\n")


def print_comparison(comparison: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("COMPARISON: Before / After UAF")
    print("="*80)
    print(comparison.to_string(index=False))
    print("="*80 + "\n")
