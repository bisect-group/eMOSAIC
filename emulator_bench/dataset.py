import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emulator_bench.common import normalize_sequence
from emulator_bench.feature_pipeline import LigandGraphStore, ProteinEmbeddingStore


def _pad_protein(embedding: np.ndarray, max_length: int):
    """Pad (or truncate) a per-residue embedding along the sequence dimension.

    Returns:
        padded_embedding: float16 [max_length, emb_dim]  (converted float32 after GPU transfer)
        mask: bool [max_length]  (1D — valid positions only; D dimension is always fully valid)
        true_length: int
    """
    length, emb_dim = embedding.shape
    if length >= max_length:
        padded = embedding[:max_length, :].astype(np.float16, copy=False)
        mask = np.ones(max_length, dtype=np.bool_)
        return padded, mask, max_length

    padded = np.zeros((max_length, emb_dim), dtype=np.float16)
    padded[:length, :] = embedding.astype(np.float16, copy=False)
    mask = np.zeros(max_length, dtype=np.bool_)
    mask[:length] = True
    return padded, mask, length


def build_protein_table(protein_store: ProteinEmbeddingStore, sequences, max_length: int, device: torch.device):
    """Stack the padded embeddings of all unique sequences on `device`.

    Returns:
        embeddings: float16 [U, max_length, emb_dim]
        masks: bool [U, max_length]
        index: normalized sequence -> row in the tables
    """
    index = {}
    for sequence in sequences:
        index.setdefault(normalize_sequence(str(sequence)), len(index))
    embeddings = masks = None
    for sequence, row in index.items():
        padded, mask, _ = _pad_protein(protein_store.get(sequence)["embedding"], max_length)
        if embeddings is None:
            embeddings = torch.zeros((len(index), max_length, padded.shape[1]), dtype=torch.float16, device=device)
            masks = torch.zeros((len(index), max_length), dtype=torch.bool, device=device)
        embeddings[row] = torch.from_numpy(padded)
        masks[row] = torch.from_numpy(mask)
    return embeddings, masks, index


class CachedEmosaicDataset(Dataset):
    """Dataset that reads precomputed protein and ligand caches.

    With `protein_index` (from `build_protein_table`) samples carry only a protein row
    index `prot_idx`; the embeddings are gathered from the device table.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        protein_store: ProteinEmbeddingStore,
        ligand_store: LigandGraphStore,
        max_length: int = 700,
        sequence_col: str = "sequence",
        smiles_col: str = "smiles",
        target_col: Optional[str] = "log10_value",
        protein_index: Optional[dict] = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.protein_store = protein_store
        self.ligand_store = ligand_store
        self.max_length = int(max_length)
        self.sequence_col = sequence_col
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.protein_idx = (
            None
            if protein_index is None
            else [protein_index[normalize_sequence(str(seq))] for seq in self.frame[sequence_col]]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        smiles = str(row[self.smiles_col])
        ligand = self.ligand_store.get(smiles)
        graph = Data(
            x=torch.from_numpy(np.asarray(ligand["x"], dtype=np.int64)),
            edge_index=torch.from_numpy(np.asarray(ligand["edge_index"], dtype=np.int64)),
            edge_attr=torch.from_numpy(np.asarray(ligand["edge_attr"], dtype=np.int64)),
        )
        if self.protein_idx is not None:
            graph.prot_idx = torch.tensor([self.protein_idx[idx]], dtype=torch.long)
        else:
            sequence = normalize_sequence(str(row[self.sequence_col]))
            padded, mask, true_length = _pad_protein(self.protein_store.get(sequence)["embedding"], self.max_length)
            graph.prot_emb = torch.from_numpy(padded).unsqueeze(0)  # (1, L, D) float16
            graph.prot_mask = torch.from_numpy(mask).unsqueeze(0)  # (1, L) bool
            graph.prot_length = torch.tensor([int(true_length)], dtype=torch.long)
        if self.target_col is not None and self.target_col in self.frame.columns:
            value = row[self.target_col]
            if pd.notna(value):
                graph.value = torch.tensor(float(value), dtype=torch.float32)
        return graph


class EmosaicDataLoader(DataLoader):
    """DataLoader that batches protein tensors alongside pyg graph batches."""

    def __init__(self, dataset: CachedEmosaicDataset, **kwargs):
        super().__init__(dataset, follow_batch=[], **kwargs)


def cat_protein_tensors(batch: Batch):
    """Concatenate per-sample protein tensors in a pyg Batch to (B, L, D) form."""
    prot_emb = torch.cat([item for item in batch.prot_emb], dim=0) if batch.prot_emb.dim() == 3 else batch.prot_emb
    prot_mask = torch.cat([item for item in batch.prot_mask], dim=0) if batch.prot_mask.dim() == 3 else batch.prot_mask
    return prot_emb, prot_mask
