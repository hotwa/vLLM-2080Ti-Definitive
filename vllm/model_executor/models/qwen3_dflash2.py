# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 draft model for Qwen3 (ported from upstream PR #52816).

DFlash2 extends DFlash v1 with two components:
- Grouped dynamic depthwise convolution around attention/MLP in each layer.
- A low-rank candidate selector that scores edges of a top-k candidate lattice
  instead of running autoregressive draft steps.

SM75 note: everything here is plain torch ops (F.pad, indexing, einsum) plus
ReplicatedLinear projections, so it runs eagerly on Turing. torch.compile and
CUDA graphs may be enabled later via the compilation config; correctness does
not depend on them.
"""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _RESIDUAL_SCALE,
    _mlp_scaled,
)
from .utils import maybe_prefix

logger = init_logger(__name__)

_FLASHINFER_TOPK_DISABLED = False


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """FlashInfer's radix top-k, or None to fall back to torch.topk.

    This top-k spans the vocabulary and is the selector's largest single cost,
    where the radix kernel is about twice torch.topk on supported hardware.
    On SM75 the radix path is unvalidated, so any failure falls back to
    torch.topk instead of disabling DFlash2.
    """
    if _FLASHINFER_TOPK_DISABLED:
        return None
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
        logger.info_once(
            "flashinfer is unavailable; the DFlash2 selector uses torch.topk."
        )
        return None
    try:
        from flashinfer import top_k
    except ImportError:
        logger.info_once(
            "flashinfer.top_k is unavailable; the DFlash2 selector uses "
            "torch.topk."
        )
        return None
    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    try:
        return impl(scores, k, sorted=True, deterministic=True)
    except Exception as exc:  # SM75 radix top-k is not guaranteed to work
        global _FLASHINFER_TOPK_DISABLED
        _FLASHINFER_TOPK_DISABLED = True
        _flashinfer_topk.cache_clear()
        logger.warning_once(
            "flashinfer.top_k failed (%s); falling back to torch.topk for the "
            "DFlash2 selector.",
            exc,
        )
        return torch.topk(scores, k, dim=-1)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Grouped dynamic depthwise convolution over flattened draft blocks.

    Pure torch implementation (no custom kernels) so it is SM75-safe under
    eager, torch.compile, and CUDA graphs.
    """
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        # Query tokens per request: the bonus token plus the mask tokens.
        block_size = 1 + speculative_config.num_speculative_tokens
        trained_block_size = draft_config.get("block_size")
        if trained_block_size is not None and int(trained_block_size) != block_size:
            logger.warning(
                "DFlash2 checkpoint was trained with block_size=%d but "
                "num_speculative_tokens=%d implies block_size=%d; acceptance "
                "rate may degrade.",
                trained_block_size,
                speculative_config.num_speculative_tokens,
                block_size,
            )
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=block_size,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._residual_scaled:
            # bf16-trained drafts exceed the fp16 range in the residual
            # stream. RMSNorm is scale-invariant, so carrying the residual
            # scaled by _RESIDUAL_SCALE (and scaling sublayer outputs before
            # each fused add+norm) is mathematically identical while staying
            # in fp16.
            s = _RESIDUAL_SCALE
            if residual is None:
                residual = hidden_states * s
                hidden_states = self.input_layernorm(residual)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states
            )
            hidden_states = self.attention_conv.finish(hidden_states * s, coefficients)

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
            hidden_states = _mlp_scaled(self.mlp, hidden_states, s)
            hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
            return hidden_states, residual

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


# Not compiled: the selector is invoked from the proposer walk with
# varying inputs, which breaks this fork's piecewise CUDA-graph replay
# input bookkeeping. It is small enough to run eagerly.
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        self.candidate_selector = CandidateSelector(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            rank=int(draft_config["selector_rank"]),
            top_k=int(draft_config["selector_top_k"]),
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "candidate_selector"),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k candidate token ids and their unary scores per position.

        Returns (ids, values), each [num_positions, selector_top_k]. ids are
        target-vocab token ids; with TP>1 the vocab shards are all-gathered
        and re-reduced so every rank produces identical candidates.
        """
        if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "DFlash2 requires an unquantized target LM head for candidate TopK."
            )

        selector = self.model.candidate_selector
        # lm_head is shared from the target model and may use a different
        # dtype (e.g. fp16 target, bf16 DFlash2 draft).
        lm_head_dtype = self.lm_head.weight.dtype
        if hidden_states.dtype != lm_head_dtype:
            hidden_states = hidden_states.to(lm_head_dtype)
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, selector.top_k)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, selector.top_k)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values


EntryClass = DFlash2Qwen3ForCausalLM
