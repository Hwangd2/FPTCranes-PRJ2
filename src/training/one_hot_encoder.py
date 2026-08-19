from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder


def one_hot_encoder() -> OneHotEncoder:
    """Construct an unknown-category-safe encoder across supported sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


__all__ = ["one_hot_encoder"]
