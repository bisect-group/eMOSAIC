import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
BAM_ROOT = CODE_ROOT / "BindingAffinityModule"
for candidate in (REPO_ROOT, CODE_ROOT, BAM_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


DEFAULT_BASE_DIR = REPO_ROOT.parents[1] / "data" / "processed" / "datasets" / "kinetic_dataset" / "ki"
DEFAULT_OUT_ROOT = REPO_ROOT.parents[1] / "outputs" / "baselines" / "eMOSAIC"
DEFAULT_EMBEDDINGS_DIR = DEFAULT_OUT_ROOT / "embeddings"
CACHE_VERSION = 3
PROTEIN_MODEL = "esmfold_v1"
DEFAULT_SEEDS = [666, 777, 888]
DEFAULT_SPLIT_GROUPS = [
    "random_splits_grouped_sequence",
    "random_splits_grouped_smiles",
    "uniprot_time_splits",
    "enzyme_sequence_splits",
    "substrate_splits",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_sequence(sequence: str) -> str:
    return (
        "".join(str(sequence).strip().upper().split())
        .replace("U", "X")
        .replace("Z", "X")
        .replace("O", "X")
        .replace("B", "X")
    )


def protein_cache_key(sequence: str, max_seq_len: int, protein_dtype: str) -> str:
    settings = f"{PROTEIN_MODEL}|{protein_dtype}|max_seq_len={max_seq_len}"
    return stable_hash(f"{settings}|{normalize_sequence(sequence)}")


def ligand_cache_key(smiles: str) -> str:
    return stable_hash(str(smiles).strip())


def protein_cache_path(embeddings_dir: Path, sequence: str, max_seq_len: int, protein_dtype: str) -> Path:
    key = protein_cache_key(sequence, max_seq_len, protein_dtype)
    return Path(embeddings_dir) / f"proteins_v{CACHE_VERSION}" / key[:2] / f"{key}.npz"


def ligand_cache_path(embeddings_dir: Path, smiles: str) -> Path:
    key = ligand_cache_key(smiles)
    return Path(embeddings_dir) / f"ligands_v{CACHE_VERSION}" / key[:2] / f"{key}.pt"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    tmp_path.replace(path)


def append_csv_row(path: Path, row: Dict) -> None:
    ensure_parent(path)
    exists = path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def _find_split_file(directory: Path, stem: str) -> Optional[Path]:
    for suffix in (".parquet", ".csv"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _split_job(split_group: str, split_name: str, directory: Path) -> Optional[Dict[str, str]]:
    paths = {stem: _find_split_file(directory, stem) for stem in ("train", "val", "test")}
    if not all(paths.values()):
        return None
    return {
        "split_group": split_group,
        "split_name": split_name,
        "train_path": str(paths["train"]),
        "val_path": str(paths["val"]),
        "test_path": str(paths["test"]),
    }


def discover_split_jobs(
    base_dir: Path,
    split_groups: Optional[Iterable[str]] = None,
    thresholds: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Find `<group>/` or `<group>/threshold_*/` folders that hold train/val/test files."""
    jobs: List[Dict[str, str]] = []
    for split_group in split_groups or DEFAULT_SPLIT_GROUPS:
        group_dir = Path(base_dir) / split_group
        if not group_dir.is_dir():
            continue
        job = _split_job(split_group, "all", group_dir)
        if job:
            jobs.append(job)
            continue
        for child in sorted(group_dir.glob("threshold_*")):
            if thresholds is not None and child.name not in thresholds:
                continue
            job = _split_job(split_group, child.name, child)
            if job:
                jobs.append(job)
    return jobs


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "r2_score": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }

    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    try:
        from scipy import stats

        spearman = float(stats.spearmanr(y_true, y_pred).statistic)
        if math.isnan(spearman):
            spearman = 0.0
    except Exception:
        true_ranks = np.argsort(np.argsort(y_true))
        pred_ranks = np.argsort(np.argsort(y_pred))
        if np.std(true_ranks) == 0 or np.std(pred_ranks) == 0:
            spearman = 0.0
        else:
            spearman = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2_score,
        "pearson": pearson,
        "spearman": spearman,
    }
