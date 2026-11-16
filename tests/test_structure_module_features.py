import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "code", ROOT / "code" / "BindingAffinityModule"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from emulator_bench import train_single_target_tvt as tvt  # noqa: E402
from emulator_bench.common import protein_cache_path  # noqa: E402
from emulator_bench.feature_pipeline import _HFEsmFoldWrapper, embed_sequences  # noqa: E402
from emulator_bench.modeling import build_model  # noqa: E402


class _CharTokenizer:
    def __call__(self, sequences, return_tensors, padding, add_special_tokens):
        ids = torch.tensor([[("ACDEFGHIKLMNPQRSTVWYX".index(c)) + 4 for c in s] for s in sequences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _tiny_esmfold():
    from transformers import EsmConfig
    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding

    trunk = {
        "num_blocks": 1,
        "sequence_state_dim": 64,
        "pairwise_state_dim": 32,
        "sequence_head_width": 32,
        "pairwise_head_width": 16,
        "structure_module": {"sequence_dim": 384, "pairwise_dim": 32, "num_blocks": 3},
    }
    config = EsmConfig(
        vocab_size=33,
        pad_token_id=1,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        is_folding_model=True,
        esmfold_config={"trunk": trunk},
    )
    torch.manual_seed(0)
    return EsmForProteinFolding(config).eval()


def test_extractor_returns_final_structure_module_state():
    model = _tiny_esmfold()
    wrapper = _HFEsmFoldWrapper(model, _CharTokenizer(), torch.device("cpu"))
    sequence = "ACDEFGHIKL"
    embedded = embed_sequences(wrapper, [sequence], torch.device("cpu"))[sequence]
    assert embedded.shape == (len(sequence), 384)
    states = model(**_CharTokenizer()([sequence], "pt", True, False))["states"]
    assert states.shape[-1] == 384
    np.testing.assert_allclose(embedded, states[-1, 0].detach().numpy(), rtol=1e-5, atol=1e-5)


def test_wrapper_refuses_1024_cache():
    tvt.check_protein_width(384, 384)
    with pytest.raises(ValueError, match="1024"):
        tvt.check_protein_width(1024, 384)


def test_protein_cache_dir_is_not_v3():
    path = protein_cache_path(Path("c"), "ACD", 700, "float16")
    assert "proteins_v4" in path.parts


def test_flatten_size_is_704_for_384():
    model = build_model(prot_input_dim=384, max_length=700)
    assert model.prot_hidden_size == 704
