"""
Uncertainty Estimation Module for Uncertainty-Aware Ensemble Framework (UAF)

Implements four complementary uncertainty estimation techniques:
1. Perplexity-based uncertainty (from token log probabilities)
2. Semantic entropy (MC1-native: Shannon entropy of softmax over choice scores)
3. Margin-based uncertainty (gap between best and 2nd-best choice score)  ← NEW
4. Haloscope (logistic regression probe on augmented hidden states)

Each method outputs a scalar uncertainty score u in [0, 1], where:
    u = 0: confident (likely correct)
    u = 1: uncertain (likely hallucinated)

Improvements vs previous version
-----------------------------------
MarginUncertainty (new)
    The gap between the best and second-best choice log-prob score is the
    single strongest per-example uncertainty signal for MC1 tasks.  A large
    gap means the model strongly prefers one answer; a small gap means it is
    hedging.  No extra forward passes required.

SemanticEntropy
    Default temperature reduced to 0.5 (was 1.0).  Lower temperature sharpens
    the softmax, amplifying the confident-vs-uncertain gap.

Haloscope (major upgrade)
    - PCA (n=64) before LR: fixes the underdetermined regime when ~82 val
      examples meet ~24k-dim hidden features (6 layers × 4096 hidden_dim).
    - Cross-validated C (grid: 0.01, 0.1, 1.0, 10.0): prevents over/under-
      regularisation without manual tuning.
    - Augmented input: hidden features are concatenated with four choice-
      distribution scalars [choice_margin_lp, choice_entropy, max_softmax,
      second_softmax].  These directly encode the model's confidence, giving
      the probe a strong shortcut signal that hidden states alone don't have.
"""

import numpy as np
import torch
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


# ============================================================================
# Utility
# ============================================================================

def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    arr     = np.array(values, dtype=np.float32)
    lo, hi  = arr.min(), arr.max()
    if hi == lo:
        return [0.0] * len(values)
    return ((arr - lo) / (hi - lo)).tolist()


# ============================================================================
# 1. Perplexity-Based Uncertainty
# ============================================================================

class PerplexityUncertainty:
    """
    Uncertainty via token-level perplexity = exp(-mean(log_probs)).
    Percentile-clipped (p2/p98) min-max normalisation: u=0 confident, u=1 uncertain.
    """

    def __init__(self):
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def compute(self, log_probs) -> float:
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        log_probs = np.array(log_probs, dtype=np.float32)
        if len(log_probs) == 0:
            return 1.0
        return float(np.exp(-np.mean(log_probs)))

    def fit(self, values: List[float]) -> None:
        arr          = np.array(values, dtype=np.float32)
        self.min_val = float(np.percentile(arr, 2))
        self.max_val = float(np.percentile(arr, 98))
        print(f"PerplexityUncertainty fitted: min={self.min_val:.4f}, max={self.max_val:.4f}")

    def normalize(self, value: float) -> float:
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit() first.")
        if self.max_val == self.min_val:
            return 0.0
        return float(np.clip((value - self.min_val) / (self.max_val - self.min_val), 0.0, 1.0))


# ============================================================================
# 2. Margin-Based Uncertainty  (NEW)
# ============================================================================

class MarginUncertainty:
    """
    Uncertainty from the gap between the best and second-best choice log-prob score.

    margin = score_best - score_second_best   (always >= 0 for log-prob scores)

    Large margin → model strongly prefers one answer → low uncertainty
    Small margin → model is hedging across choices → high uncertainty

    After percentile-clipped normalization the margin is INVERTED so that
    the output u follows the convention u=0 (confident) / u=1 (uncertain).

    Why this works for TruthfulQA MC1
    -----------------------------------
    TruthfulQA questions are designed so that truthful answers have a
    distinctive linguistic signature.  When a model knows the answer, it
    assigns a much higher log-prob to the correct choice than to distractors.
    The margin directly captures this.  Empirically, margin-based AUROC on
    hallucination detection tasks often exceeds entropy-based AUROC.
    """

    def __init__(self):
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def compute(self, choice_scores: List[float]) -> float:
        """
        Raw margin: best_score - second_best_score (in log-prob space).
        Returns 0.0 if fewer than 2 choices.
        """
        if len(choice_scores) < 2:
            return 0.0
        ss = sorted(choice_scores, reverse=True)
        return float(ss[0] - ss[1])

    def fit(self, values: List[float]) -> None:
        """Fit normalization from a list of raw margin values."""
        arr          = np.array(values, dtype=np.float32)
        self.min_val = float(np.percentile(arr, 2))
        self.max_val = float(np.percentile(arr, 98))
        print(f"MarginUncertainty fitted: min={self.min_val:.4f}, max={self.max_val:.4f}")

    def normalize(self, value: float) -> float:
        """Normalize and invert: high margin → u near 0, low margin → u near 1."""
        if self.min_val is None or self.max_val is None:
            raise ValueError("Call fit() first.")
        if self.max_val == self.min_val:
            return 0.5
        norm = float(np.clip((value - self.min_val) / (self.max_val - self.min_val), 0.0, 1.0))
        return 1.0 - norm   # invert: confident (large margin) → low uncertainty


# ============================================================================
# 3. Semantic Entropy (MC1-native)
# ============================================================================

class SemanticEntropy:
    """
    Uncertainty via Shannon entropy of softmax over choice log-prob scores.

    temperature=0.5 (default)
    --------------------------
    Lower temperature sharpens the softmax before computing entropy.
    When a model strongly prefers one choice, temperature<1 makes that
    preference even more pronounced → very low entropy.
    When probability mass is spread evenly, entropy stays high regardless.
    This amplifies the discriminative signal without changing relative rankings.

    Both val and test phases must use the same temperature so that the
    global normalisation range (semantic_min / semantic_max) fitted on val
    remains valid on the test set.
    """

    def __init__(self, threshold: float = 0.85):
        self.threshold  = threshold
        self.epsilon    = 1e-10
        self._st_model  = None

    def compute_from_choice_scores(
        self,
        choice_scores: List[float],
        temperature: float = 0.5,
    ) -> float:
        """
        Shannon entropy H = -sum(p * log(p)) of softmax(choice_scores / temperature).

        Args:
            choice_scores: mean log-prob per choice (from infer_example)
            temperature:   softmax sharpness (0.5 is the recommended default)

        Returns:
            float: raw entropy (unnormalized)
        """
        if len(choice_scores) <= 1:
            return 0.0
        arr   = np.array(choice_scores, dtype=np.float64)
        arr   = arr / max(temperature, 1e-6)
        arr  -= arr.max()
        probs = np.exp(arr)
        probs /= probs.sum()
        return float(-np.sum(probs * np.log(probs + self.epsilon)))

    def compute(self, responses: List[str]) -> float:
        """Fallback for open-ended text. Not used for MC1."""
        if len(responses) <= 1:
            return 0.0
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        from sklearn.cluster import AgglomerativeClustering
        emb  = self._st_model.encode(responses, convert_to_tensor=False)
        emb  = np.array(emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + self.epsilon
        emb   = emb / norms
        clust = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - self.threshold,
            metric="cosine", linkage="average"
        )
        labels = clust.fit_predict(emb)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(responses)
        return float(-np.sum(probs * np.log(probs + self.epsilon)))

    def normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.0
        return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


# ============================================================================
# 4. Haloscope — Augmented features + PCA + cross-validated LR
# ============================================================================

class Haloscope:
    """
    Uncertainty via logistic regression on augmented hidden-state features.

    Input features per example (concatenated before fitting)
    ---------------------------------------------------------
    hidden_features   : mean-pooled hidden states from last num_layers layers.
                        Shape: (num_layers × hidden_dim,) — e.g. (6 × 4096,)
    choice_margin_lp  : gap between best and 2nd-best log-prob choice score.
    choice_entropy    : Shannon entropy of softmax over choice scores.
    max_softmax_prob  : probability assigned to the predicted choice.
    second_softmax    : probability assigned to the second-best choice.

    The 4 scalar augmentation features directly encode the model's confidence
    distribution.  They are highly predictive of hallucination and give the LR
    probe a strong shortcut that hidden states alone don't have.

    Dimensionality reduction (PCA → 64 dims)
    -----------------------------------------
    With ~82 val examples and up to 24 576-dim input (6 × 4096), logistic
    regression is grotesquely underdetermined without reduction.  PCA retains
    the directions of maximum variance — which correlate most with correctness
    — while regularising the geometry.  StandardScaler is applied before PCA
    so that no single layer dominates the PCA directions.

    Cross-validated C selection
    ----------------------------
    Tries C ∈ {0.01, 0.1, 1.0, 10.0} with 3-fold CV on AUROC.
    Prevents over/under-regularisation without manual tuning.

    Output
    ------
    predict() returns P(wrong/hallucinated) ∈ [0, 1].
    """

    # Expected number of augmentation scalars appended to hidden features.
    N_AUG = 4   # [choice_margin_lp, choice_entropy, max_softmax, second_softmax]

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        n_pca_components: int = 64,
    ):
        self.C                = C
        self.max_iter         = max_iter
        self.n_pca_components = n_pca_components
        self.scaler           = StandardScaler()
        self.pca: Optional[PCA] = None
        self.model            = LogisticRegression(
            C=C, max_iter=max_iter, random_state=42, solver="lbfgs"
        )
        self.is_fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def build_augmented_features(
        hidden: np.ndarray,
        choice_margin_lp: float,
        choice_entropy: float,
        max_softmax: float,
        second_softmax: float,
    ) -> np.ndarray:
        """
        Concatenate hidden features with the 4 choice-distribution scalars.

        Call this from main.py both when building X_train for fit() and when
        building the feature vector for predict().  Consistent augmentation in
        both phases is critical.

        Args:
            hidden           : shape (hidden_dim,) — mean-pooled hidden states
            choice_margin_lp : best_score - second_best_score (log-prob)
            choice_entropy   : Shannon entropy of softmax over choice scores
            max_softmax      : max softmax prob over choices
            second_softmax   : second-highest softmax prob

        Returns:
            np.ndarray shape (hidden_dim + 4,)
        """
        extra = np.array(
            [choice_margin_lp, choice_entropy, max_softmax, second_softmax],
            dtype=np.float32,
        )
        return np.concatenate([hidden.astype(np.float32), extra])

    def _reduce(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """StandardScale then PCA-reduce.  fit=True during training."""
        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        if fit:
            n_comp = min(self.n_pca_components, X.shape[0] - 1, X.shape[1])
            if n_comp > 0 and n_comp < X.shape[1]:
                self.pca = PCA(n_components=n_comp, random_state=42)
                return self.pca.fit_transform(X_scaled)
            self.pca = None
            return X_scaled

        # Inference must reuse the PCA fitted on training data even when
        # predicting one sample (X.shape[0] == 1).
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit with scaling, PCA, and cross-validated C.

        Args:
            X: shape (n_samples, n_features) — augmented hidden features
            y: shape (n_samples,) — 1 = wrong/hallucinated, 0 = correct
        """
        assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        X_red    = self._reduce(X, fit=True)
        n_folds  = min(3, len(np.unique(y)))
        best_c   = self.C
        best_cv  = -1.0

        if n_folds >= 2 and len(X_red) >= 2 * n_folds:
            for c in [0.01, 0.1, 1.0, 10.0]:
                lr = LogisticRegression(
                    C=c, max_iter=self.max_iter, random_state=42, solver="lbfgs"
                )
                try:
                    scores = cross_val_score(lr, X_red, y, cv=n_folds, scoring="roc_auc")
                    mean_s = float(scores.mean())
                    if mean_s > best_cv:
                        best_cv = mean_s
                        best_c  = c
                except Exception:
                    pass
            print(f"  Haloscope CV: best C={best_c}  AUROC={best_cv:.4f}")
        else:
            print(f"  Haloscope: insufficient samples for CV, using C={best_c}")

        self.model = LogisticRegression(
            C=best_c, max_iter=self.max_iter, random_state=42, solver="lbfgs"
        )
        self.model.fit(X_red, y)
        self.is_fitted = True
        print(
            f"Haloscope fitted: n={len(X)}, "
            f"raw_dim={X.shape[1]}, pca_dim={X_red.shape[1]}, "
            f"classes={np.unique(y).tolist()}, C={best_c}"
        )

    def predict(self, X: np.ndarray) -> float:
        """
        Predict P(hallucination) for a single example.

        Args:
            X: shape (n_features,) or (1, n_features)

        Returns:
            float in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_red = self._reduce(X, fit=False)
        return float(self.model.predict_proba(X_red)[0, 1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict P(hallucination) for a batch of examples."""
        if not self.is_fitted:
            raise ValueError("Call fit() first.")
        X     = np.array(X, dtype=np.float32)
        X_red = self._reduce(X, fit=False)
        return self.model.predict_proba(X_red)[:, 1]
