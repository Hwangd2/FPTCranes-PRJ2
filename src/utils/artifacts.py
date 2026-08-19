from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


@lru_cache(maxsize=128)
def _read_csv(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path)


@lru_cache(maxsize=128)
def _read_json(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


@lru_cache(maxsize=4)
def _load_model(path: str, modified_ns: int) -> Any:
    del modified_ns
    return joblib.load(path)


def load_csv(path: Path) -> pd.DataFrame:
    """Load an optional CSV and invalidate the cache when the file changes."""
    if not path.is_file():
        return pd.DataFrame()
    resolved = path.resolve()
    return _read_csv(str(resolved), resolved.stat().st_mtime_ns)


def load_json(path: Path) -> dict[str, Any]:
    """Load an optional JSON object and invalidate the cache when it changes."""
    if not path.is_file():
        return {}
    resolved = path.resolve()
    return _read_json(str(resolved), resolved.stat().st_mtime_ns)


def load_model(path: Path) -> Any:
    """Load a required serialized model with file-aware resource caching."""
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = path.resolve()
    return _load_model(str(resolved), resolved.stat().st_mtime_ns)
