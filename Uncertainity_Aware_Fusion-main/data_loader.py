"""
TruthfulQA Data Loader for Uncertainty-Aware Ensemble Framework (UAF)

This module provides deterministic data loading and preprocessing for the TruthfulQA
multiple_choice dataset (validation split). It ensures reproducibility and provides
utilities for prompt formatting and evaluation across model ensemble pipelines.

Usage:
    from data_loader import load_truthfulqa, extract_mc1, format_prompt, evaluate_mc1
    
    val_set, test_set = load_truthfulqa(seed=42)
    for example in val_set:
        question, choices, correct_idx = extract_mc1(example)
        prompt = format_prompt(question, choices[0])
        ...
"""

from typing import List, Tuple, Any
from datasets import load_dataset


def load_truthfulqa(seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Load and split the TruthfulQA multiple_choice validation dataset.
    
    Loads the "validation" split from HuggingFace's truthful_qa dataset,
    deterministically shuffles it, and splits into validation (10%) and
    test (90%) sets.
    
    Args:
        seed: Random seed for deterministic shuffling. Default: 42
        
    Returns:
        Tuple[List[dict], List[dict]]: (val_set, test_set)
            - val_set (10%): ~82 examples for selector component
            - test_set (90%): ~735 examples for fuser component
            
    Raises:
        DatasetNotFoundError: If dataset cannot be loaded from HuggingFace
    """
    # Load dataset
    dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    # Convert to list for deterministic shuffling
    data = list(dataset)
    total_size = len(data)
    print(f"Dataset loaded: {total_size} examples")
    
    # Deterministic shuffle
    rng = __import__("random").Random(seed)
    rng.shuffle(data)
    
    # Split: 10% validation, 90% test
    val_size = max(1, int(0.10 * total_size))  # Ensure at least 1 example
    
    val_set = data[:val_size]
    test_set = data[val_size:]
    
    print(f"Validation set: {len(val_set)} examples (10%)")
    print(f"Test set: {len(test_set)} examples (90%)")
    
    return val_set, test_set


def extract_mc1(example: dict) -> Tuple[str, List[str], int]:
    """
    Extract question, choices, and correct answer index from an example.
    
    Args:
        example: Dictionary containing "question" and "mc1_targets" fields
        
    Returns:
        Tuple[str, List[str], int]: (question, choices, correct_idx)
            - question: The question text
            - choices: List of answer choices
            - correct_idx: Index of the correct answer (where label == 1)
            
    Raises:
        AssertionError: If no label or multiple labels are set to 1
    """
    question: str = example["question"]
    targets: dict = example["mc1_targets"]
    choices: List[str] = targets["choices"]
    labels: List[int] = targets["labels"]
    
    # Validate: exactly one label must be 1
    correct_indices = [i for i, label in enumerate(labels) if label == 1]
    assert len(correct_indices) == 1, (
        f"Expected exactly one correct label (label == 1), "
        f"but found {len(correct_indices)}: {correct_indices}"
    )
    
    correct_idx: int = correct_indices[0]
    
    return question, choices, correct_idx


def format_prompt(question: str, choice: str) -> str:
    """
    Format a question and choice into a standardized prompt.
    
    Ensures consistent prompt formatting across all models in the ensemble
    for fair and reproducible inference.
    
    Args:
        question: The question text
        choice: The answer choice text
        
    Returns:
        str: Formatted prompt in the form "Q: {question}\nA: {choice}"
    """
    return f"Q: {question}\nA: {choice}"


def evaluate_mc1(example: dict, predicted_idx: int) -> int:
    """
    Evaluate a prediction against the ground truth multiple-choice answer.
    
    Args:
        example: Dictionary containing "mc1_targets" with ground truth labels
        predicted_idx: Index of the predicted answer choice
        
    Returns:
        int: 1 if prediction is correct, 0 otherwise
    """
    targets: dict = example["mc1_targets"]
    labels: List[int] = targets["labels"]
    
    # Correct if predicted choice has label == 1
    is_correct: int = int(labels[predicted_idx] == 1)
    
    return is_correct
