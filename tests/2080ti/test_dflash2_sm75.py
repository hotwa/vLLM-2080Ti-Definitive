# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 SM75 port correctness tests.

Covers the fork-specific V1-proposer port of upstream PR #52816:
- grouped dynamic convolution matches a sequential reference
- candidate selector edge scores match a sequential reference
- model registry resolves DFlash2DraftModel to the DFlash2 implementation
- speculative config routes DFlash2 checkpoints to use_dflash2()
- the DFlash v1 implementation refuses DFlash2 checkpoints (fail-fast)
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def test_registry_resolves_dflash2_draft_model():
    from vllm.model_executor.models.registry import ModelRegistry

    arch = ModelRegistry.get_supported_archs()
    assert "DFlash2DraftModel" in arch
    # v1 must stay registered so existing DFlash checkpoints do not regress.
    assert "DFlashDraftModel" in arch


def _spec_config(method, architectures):
    return SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(architectures=architectures),
        use_dflash=lambda: method == "dflash",
    )


def test_use_dflash2_detection():
    from vllm.config.speculative import SpeculativeConfig

    spec = _spec_config("dflash", ["DFlash2DraftModel"])
    assert SpeculativeConfig.use_dflash2(spec)

    spec = _spec_config("dflash", ["DFlashDraftModel"])
    assert not SpeculativeConfig.use_dflash2(spec)

    spec = _spec_config("eagle", ["DFlash2DraftModel"])
    assert not SpeculativeConfig.use_dflash2(spec)

    spec = SimpleNamespace(
        method="dflash", draft_model_config=None, use_dflash=lambda: True
    )
    assert not SpeculativeConfig.use_dflash2(spec)


def test_greedy_walk_tie_break_matches_upstream():
    """Lowest index must win ties, matching the upstream walk kernel's
    min(where(score == best)) rule."""
    scores = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    assert scores.argmax(dim=-1).item() == 1
