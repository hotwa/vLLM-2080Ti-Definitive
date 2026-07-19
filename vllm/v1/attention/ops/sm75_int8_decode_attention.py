# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SM75 INT8 paged-attention prototype for decode and MTP verify."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

PART_TOKENS = 32
QUERY_HEADS = 8
KV_HEADS = 1
HEAD_SIZE = 256
MAX_QUERY_TOKENS = 4
_WORKSPACES: dict[str, tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}


@lru_cache(maxsize=1)
def _extension():
    source_root = Path(__file__).resolve().parents[4]
    build_directory = Path(
        os.getenv("VLLM_SM75_INT8_BUILD_DIR", "/tmp/vllm-sm75-int8-decode")
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="vllm_sm75_int8_decode_attention",
        sources=[str(source_root / "csrc" / "sm75_int8_decode_attention.cu")],
        build_directory=str(build_directory),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=os.getenv("VLLM_SM75_INT8_BUILD_VERBOSE", "0") == "1",
    )


def workspace(
    max_context_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_parts = (max_context_tokens + PART_TOKENS - 1) // PART_TOKENS
    partial_output = torch.empty(
        (num_parts, MAX_QUERY_TOKENS, QUERY_HEADS, HEAD_SIZE),
        dtype=torch.float32,
        device=device,
    )
    partial_max = torch.empty(
        (num_parts, MAX_QUERY_TOKENS, QUERY_HEADS),
        dtype=torch.float32,
        device=device,
    )
    partial_sum = torch.empty_like(partial_max)
    return partial_output, partial_max, partial_sum


def get_workspace(
    min_context_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a process-global, grow-only workspace for one CUDA device.

    Attention layers execute serially on the worker's current stream, so sharing
    avoids reserving one large partial-output buffer per model layer.
    """
    device = torch.device(device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = str(device)
    required_parts = (min_context_tokens + PART_TOKENS - 1) // PART_TOKENS
    capacity_parts = 1 << max(0, required_parts - 1).bit_length()
    cached = _WORKSPACES.get(key)
    if cached is None or cached[0] < capacity_parts:
        capacity_tokens = capacity_parts * PART_TOKENS
        cached = (capacity_parts, workspace(capacity_tokens, device))
        _WORKSPACES[key] = cached
    return cached[1]


def attention_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    block_table: torch.Tensor,
    workspaces: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    seq_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    partial_output, partial_max, partial_sum = workspaces
    _extension().attention_out(
        query,
        key,
        value,
        key_scale,
        value_scale,
        block_table,
        partial_output,
        partial_max,
        partial_sum,
        output,
        seq_len,
        softmax_scale,
    )
    return output
