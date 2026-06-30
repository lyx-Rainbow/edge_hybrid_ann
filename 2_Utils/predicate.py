"""
Shared predicate evaluation for complex-predicate queries.

Supports Polish-notation boolean expressions:
  "AND 0 1"        — vector has BOTH label 0 AND label 1
  "OR 0 1"         — vector has EITHER label 0 OR label 1
  "NOT 0"          — vector does NOT have label 0
  "AND 0 NOT 1"    — vector has label 0 AND NOT label 1
  "OR 0 OR 1 2"    — vector has label 0 OR 1 OR 2
"""

import numpy as np


def evaluate_predicate(tokens: list[str], mds: list[int]) -> bool:
    """Evaluate a boolean formula (tokenized Polish notation) against one
    vector's label list."""
    stack = []
    for token in reversed(tokens):
        if token in ("AND", "OR", "NOT"):
            if token == "AND":
                a, b = stack.pop(), stack.pop()
                stack.append(a and b)
            elif token == "OR":
                a, b = stack.pop(), stack.pop()
                stack.append(a or b)
            else:  # NOT
                stack.append(not stack.pop())
        else:
            stack.append(int(token) in mds)
    return stack[0]


def compute_qualified_indices(formula: str, train_mds: list) -> np.ndarray:
    """Return int32 array of training indices that satisfy the formula."""
    tokens = formula.split()
    qualified = [
        i for i, md in enumerate(train_mds)
        if md and evaluate_predicate(tokens, md)
    ]
    return np.array(qualified, dtype=np.int32)


def compute_filter_selectivity(formula: str, train_mds: list) -> float:
    """Fraction of training vectors satisfying the formula."""
    qualified = compute_qualified_indices(formula, train_mds)
    return len(qualified) / len(train_mds)


def build_inverted_index(train_mds: list, n_labels: int) -> dict:
    """label -> sorted int32 array of training indices."""
    label_to_indices = {i: [] for i in range(n_labels)}
    for idx, labels in enumerate(train_mds):
        for lab in labels:
            label_to_indices[lab].append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in label_to_indices.items()}
