"""Equivalence of the emulator_bench wrapper with the authors' BindingAffinityModule code.

Run from baselines/eMOSAIC on CPU:  CUDA_VISIBLE_DEVICES= python -m pytest tests -q
"""
import copy
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PygLoader

ROOT = Path(__file__).resolve().parents[1]
BAM = ROOT / "code" / "BindingAffinityModule"
for path in (ROOT, ROOT / "code", BAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from emulator_bench import train_single_target_tvt as tvt  # noqa: E402
from emulator_bench.common import ligand_cache_path, protein_cache_path, set_seed  # noqa: E402
from emulator_bench.dataset import CachedEmosaicDataset, EmosaicDataLoader, build_protein_table  # noqa: E402
from emulator_bench.feature_pipeline import (  # noqa: E402
    LigandGraphStore,
    ProteinEmbeddingStore,
    ligand_cache_item,
    protein_cache_item,
    save_ligand_pt,
    save_protein_npz,
)
from emulator_bench.modeling import build_model  # noqa: E402
from model import BindingModel2  # noqa: E402
import trainer as original_trainer  # noqa: E402

MAX_LEN, PROT_DIM, EMB_DIM = 700, 384, 300
SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC", "OC(=O)CC(O)(CC(O)=O)C(O)=O", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"]
LENGTHS = [30, 75, 120, 700, 10, 400]
LR, WEIGHT_DECAY = 1e-4, 1e-4


def _load_original_collate():
    spec = importlib.util.spec_from_file_location("original_dataset", BAM / "dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collate_fn


original_collate = _load_original_collate()


def _random_sequences(lengths, seed=0):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), n)) for n in lengths]


def _embeddings(sequences, seed=1):
    """float16-representable per-residue embeddings, as stored in the cache."""
    rng = np.random.default_rng(seed)
    return {s: rng.standard_normal((len(s), PROT_DIM)).astype(np.float16) for s in sequences}


class _Stub:
    def __init__(self, items):
        self.items = items

    def get(self, key):
        return self.items[key]


def _original_batch(sequences, embeddings, smiles, targets):
    rows = [
        (torch.from_numpy(embeddings[s].astype(np.float32)), m, torch.tensor(t, dtype=torch.float), "P", torch.tensor(i))
        for i, (s, m, t) in enumerate(zip(sequences, smiles, targets))
    ]
    return original_collate(rows, MAX_LEN, {})


def _wrapper_batch(sequences, embeddings, smiles, targets, use_table):
    frame = pd.DataFrame({"sequence": sequences, "smiles": smiles, "log10_value": targets})
    protein_store = _Stub({s: {"embedding": e} for s, e in embeddings.items()})
    ligand_store = _Stub({m: ligand_cache_item(m) for m in smiles})
    table = index = None
    if use_table:
        *table, index = build_protein_table(protein_store, sequences, MAX_LEN, torch.device("cpu"))
    dataset = CachedEmosaicDataset(frame, protein_store, ligand_store, MAX_LEN, protein_index=index)
    batch = next(iter(EmosaicDataLoader(dataset, batch_size=len(frame), shuffle=False)))
    return tvt._prepare_batch(batch, torch.device("cpu"), table)


@pytest.fixture(scope="module")
def data():
    sequences = _random_sequences(LENGTHS)
    embeddings = _embeddings(sequences)
    targets = np.random.default_rng(2).uniform(-2, 3, len(sequences)).round(3).tolist()
    return sequences, embeddings, SMILES, targets


def _new_original():
    return BindingModel2(2, EMB_DIM, 0.2, "gin", EMB_DIM + 704)


def _randomized_original(seed=0):
    """Original model with a non-degenerate attention matrix and BatchNorm running statistics."""
    set_seed(seed)
    model = _new_original()
    torch.nn.init.normal_(model.attention.param, std=0.05)
    return model


def _double_batch(batch):
    batch.prot_emb = batch.prot_emb.double()
    return batch


def _original_forward(model, ob):
    protein, _, mask, graphs, targets, _, _ = ob
    return model(protein.double(), graphs, mask.double()), targets.double()


def _assert_same_params(a, b, attr, atol, rtol=0):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    for name in pa:
        x, y = getattr(pa[name], attr), getattr(pb[name], attr)
        assert x is not None and y is not None, name
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=lambda m, n=name: f"{n}: {m}")


def _paired_models(data):
    """Original and wrapper models in fp64 with identical weights (and BN statistics)."""
    sequences, embeddings, smiles, targets = data
    original = _randomized_original().double()
    ob = _original_batch(sequences, embeddings, smiles, targets)
    original.train()
    with torch.no_grad():
        for _ in range(3):
            _original_forward(original, ob)
    wrapper = build_model(max_length=MAX_LEN, prot_input_dim=PROT_DIM).double()
    assert original.state_dict().keys() == wrapper.state_dict().keys()
    wrapper.load_state_dict(original.state_dict(), strict=True)
    wb = _double_batch(_wrapper_batch(sequences, embeddings, smiles, targets, use_table=True))
    return original, wrapper, ob, wb


def test_wrapper_flatten_size_matches_original_attention_size():
    wrapper = build_model(max_length=MAX_LEN, prot_input_dim=PROT_DIM)
    assert wrapper.prot_hidden_size == _new_original().attention.prot_hidden_size == 704


@pytest.mark.parametrize("mode", ["eval", "train_seeded_dropout"])
def test_outputs_loss_and_gradients_match_original(data, mode):
    original, wrapper, ob, wb = _paired_models(data)
    for model in (original, wrapper):
        model.eval() if mode == "eval" else model.train()
    torch.manual_seed(7)
    out_o, target = _original_forward(original, ob)
    torch.manual_seed(7)
    out_w = wrapper(wb)
    loss_o, loss_w = F.mse_loss(out_o, target), F.mse_loss(out_w, wb.value.double())
    torch.testing.assert_close(wb.value.double(), target)
    torch.testing.assert_close(out_w, out_o, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(loss_w, loss_o, atol=1e-10, rtol=1e-10)
    loss_o.backward()
    loss_w.backward()
    _assert_same_params(original, wrapper, "grad", atol=1e-10, rtol=1e-8)
    if mode == "train_seeded_dropout":
        for (k, a), (_, b) in zip(original.state_dict().items(), wrapper.state_dict().items()):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10, msg=lambda m, k=k: f"{k}: {m}")


@pytest.mark.parametrize("use_table", [True, False])
def test_batches_equal_original_collate(data, use_table):
    sequences, embeddings, smiles, targets = data
    long_seq = _random_sequences([720], seed=5)[0]
    sequences = sequences + [long_seq]
    embeddings = {**embeddings, **_embeddings([long_seq], seed=6)}
    smiles = smiles + ["CCCC"]
    targets = targets + [0.5]
    protein, _, mask, graphs, y, _, _ = _original_batch(sequences, embeddings, smiles, targets)
    wb = _wrapper_batch(sequences, embeddings, smiles, targets, use_table)
    assert torch.equal(wb.prot_emb, protein)
    assert torch.equal(wb.prot_mask, mask[:, :, 0].bool())
    assert torch.equal(mask, wb.prot_mask.unsqueeze(-1).expand_as(mask).float())
    assert torch.equal(wb.value, y)
    for key in ("x", "edge_index", "edge_attr", "batch"):
        assert torch.equal(getattr(wb, key), getattr(graphs, key)), key


def _make_cache_and_splits(tmp_path, data):
    sequences, embeddings, smiles, targets = data
    cache = tmp_path / "cache"
    for s in sequences:
        path = protein_cache_path(cache, s, MAX_LEN, "float16")
        save_protein_npz(path, protein_cache_item(s, embeddings[s], "float16"))
    for m in smiles:
        save_ligand_pt(ligand_cache_path(cache, m), ligand_cache_item(m))
    frame = pd.DataFrame({"sequence": sequences, "smiles": smiles, "log10_value": targets})
    split = tmp_path / "split.csv"
    frame.to_csv(split, index=False)
    return cache, split


def test_one_optimizer_step_matches_original_training_loop(data, tmp_path, monkeypatch):
    """Runs the wrapper's real main() for one epoch and the original train() for one step."""
    sequences, embeddings, smiles, targets = data
    cache, split = _make_cache_and_splits(tmp_path, data)
    out = tmp_path / "run"
    seed = 123
    argv = [
        "train", "--train_path", str(split), "--val_path", str(split), "--test_path", str(split),
        "--embeddings_dir", str(cache), "--out_dir", str(out), "--epochs", "1", "--batch_size", str(len(sequences)),
        "--num_workers", "0", "--device", "cpu", "--seed", str(seed),
    ]
    # Dropout masks depend on RNG consumption order, so remove them from both sides.
    monkeypatch.setattr(F, "dropout", lambda input, p=0.5, training=True, inplace=False: input)
    # Keep the sample order identical to the original single batch.
    monkeypatch.setattr(tvt, "EmosaicDataLoader", lambda ds, **kw: EmosaicDataLoader(ds, **{**kw, "shuffle": False}))
    monkeypatch.setattr(sys, "argv", argv)
    tvt.main()
    wrapper_state = torch.load(out / "checkpoint_last.pt", map_location="cpu")["model_state_dict"]

    set_seed(seed)
    original = _new_original()
    optimizer = torch.optim.AdamW(original.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
    ob = _original_batch(sequences, embeddings, smiles, targets)
    # The original metric code passes `squared=`, which current scikit-learn removed; metrics run after the step.
    monkeypatch.setattr(
        original_trainer, "mean_squared_error", lambda a, b, squared=True: float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    )
    original_trainer.train(original, [ob], optimizer, torch.nn.MSELoss(), torch.device("cpu"))

    original_state = original.state_dict()
    assert original_state.keys() == wrapper_state.keys()
    for name, value in original_state.items():
        torch.testing.assert_close(wrapper_state[name], value, atol=1e-6, rtol=1e-5, msg=lambda m, n=name: f"{n}: {m}")


def test_neither_original_nor_wrapper_clips_gradients():
    for path in (BAM / "trainer.py", BAM / "main.py", ROOT / "emulator_bench" / "train_single_target_tvt.py"):
        assert "clip_grad" not in path.read_text(), path


def test_lr_schedule_matches_original():
    def lrs(step):
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        values = [opt.param_groups[0]["lr"]]
        for epoch in range(1, 51):
            step(sched, epoch)
            values.append(opt.param_groups[0]["lr"])
        return values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original = lrs(lambda s, epoch: s.step(epoch))
    wrapper = lrs(lambda s, epoch: s.step())
    np.testing.assert_allclose(wrapper, original, rtol=1e-12)
