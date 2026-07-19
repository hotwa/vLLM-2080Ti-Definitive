# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the SM75 INT8 per-token-head paged attention path.

The default shape matches one TP rank of Ornith/Qwen3.5 MoE 35B on TP=2:
8 query heads, 1 KV head, and head dimension 256.  The benchmark exercises
the same ``unified_attention`` entry point used by the Triton backend and
compares short contexts against an explicit PyTorch dequantized reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode


@dataclass
class Result:
    context_tokens: int
    median_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second_scanned: float
    max_abs_error: float | None
    mean_abs_error: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="128,1024,5000,38000,80000")
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument(
        "--permute-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference-max-context", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--backend",
        choices=("triton", "sm75-cuda", "sm75-integrated"),
        default="triton",
    )
    return parser.parse_args()


def quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.float().abs().amax(dim=-1).clamp_min(1e-8).div(127.0)
    quantized = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127)
    return quantized.to(torch.int8), scale


def make_case(
    context_tokens: int,
    query_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    permute_blocks: bool,
    device: torch.device,
) -> dict[str, torch.Tensor | float | int]:
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")

    num_blocks = math.ceil(context_tokens / block_size)
    padded_tokens = num_blocks * block_size

    if not 1 <= query_tokens <= 4:
        raise ValueError("query_tokens must be in [1, 4]")
    if context_tokens < query_tokens:
        raise ValueError("context_tokens must cover all query tokens")
    query = torch.randn(
        (query_tokens, query_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    key_fp = torch.randn(
        (padded_tokens, kv_heads, head_size), dtype=torch.float16, device=device
    )
    value_fp = torch.randn_like(key_fp)
    key, key_scale = quantize_per_token_head(key_fp)
    value, value_scale = quantize_per_token_head(value_fp)

    key = key.view(num_blocks, block_size, kv_heads, head_size)
    value = value.view(num_blocks, block_size, kv_heads, head_size)
    key_scale = key_scale.view(num_blocks, block_size, kv_heads)
    value_scale = value_scale.view(num_blocks, block_size, kv_heads)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    if permute_blocks and num_blocks > 1:
        logical_to_physical = torch.randperm(
            num_blocks, dtype=torch.int32, device=device
        )
        key_physical = torch.empty_like(key)
        value_physical = torch.empty_like(value)
        key_scale_physical = torch.empty_like(key_scale)
        value_scale_physical = torch.empty_like(value_scale)
        key_physical[logical_to_physical.long()] = key
        value_physical[logical_to_physical.long()] = value
        key_scale_physical[logical_to_physical.long()] = key_scale
        value_scale_physical[logical_to_physical.long()] = value_scale
        key = key_physical
        value = value_physical
        key_scale = key_scale_physical
        value_scale = value_scale_physical
        block_table = logical_to_physical

    return {
        "query": query,
        "key": key,
        "value": value,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "output": torch.empty_like(query),
        "cu_seqlens_q": torch.tensor(
            [0, query_tokens], dtype=torch.int32, device=device
        ),
        "seqused_k": torch.tensor([context_tokens], dtype=torch.int32, device=device),
        "block_table": block_table.unsqueeze(0),
        "softmax_scale": head_size**-0.5,
        "context_tokens": context_tokens,
        "query_tokens": query_tokens,
    }


def run_attention(
    case: dict[str, object],
    backend: str,
) -> torch.Tensor:
    query = case["query"]
    output = case["output"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(output, torch.Tensor)

    if backend == "sm75-integrated":
        impl = case["impl"]
        metadata = case["metadata"]
        used = impl._try_sm75_int8_decode(
            query,
            output,
            case["key"],
            case["value"],
            case["key_scale"],
            case["value_scale"],
            metadata,
            int(case["query_tokens"]),
            None,
            None,
        )
        if not used:
            raise RuntimeError("SM75 integration guard did not select the kernel")
        return output

    if backend == "sm75-cuda":
        from vllm.v1.attention.ops.sm75_int8_decode_attention import attention_out

        workspaces = case["workspaces"]
        assert isinstance(workspaces, tuple)
        return attention_out(
            query,
            case["key"],
            case["value"],
            case["key_scale"],
            case["value_scale"],
            case["block_table"],
            workspaces,
            output,
            int(case["context_tokens"]),
            float(case["softmax_scale"]),
        )

    unified_attention(
        q=query,
        k=case["key"],
        v=case["value"],
        out=output,
        cu_seqlens_q=case["cu_seqlens_q"],
        max_seqlen_q=int(case["query_tokens"]),
        seqused_k=case["seqused_k"],
        max_seqlen_k=case["context_tokens"],
        softmax_scale=case["softmax_scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=case["block_table"],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=case["key_scale"],
        v_scale_cache=case["value_scale"],
    )
    return output


def reference_attention(
    case: dict[str, object],
) -> torch.Tensor:
    context_tokens = int(case["context_tokens"])
    query = case["query"]
    key = case["key"]
    value = case["value"]
    key_scale = case["key_scale"]
    value_scale = case["value_scale"]
    assert isinstance(query, torch.Tensor)
    assert isinstance(key, torch.Tensor)
    assert isinstance(value, torch.Tensor)
    assert isinstance(key_scale, torch.Tensor)
    assert isinstance(value_scale, torch.Tensor)

    kv_heads = key.shape[2]
    repeats = query.shape[1] // kv_heads
    head_size = query.shape[-1]
    block_table = case["block_table"]
    assert isinstance(block_table, torch.Tensor)
    logical_blocks = block_table[0].long()
    key_logical = key.index_select(0, logical_blocks)
    value_logical = value.index_select(0, logical_blocks)
    key_scale_logical = key_scale.index_select(0, logical_blocks)
    value_scale_logical = value_scale.index_select(0, logical_blocks)
    key_dequant = (
        key_logical.flatten(0, 1)[:context_tokens].float()
        * key_scale_logical.flatten(0, 1)[:context_tokens].unsqueeze(-1)
    ).repeat_interleave(repeats, dim=1)
    value_dequant = (
        value_logical.flatten(0, 1)[:context_tokens].float()
        * value_scale_logical.flatten(0, 1)[:context_tokens].unsqueeze(-1)
    ).repeat_interleave(repeats, dim=1)
    outputs = []
    query_tokens = query.shape[0]
    for query_token in range(query_tokens):
        query_seq_len = context_tokens - (query_tokens - 1 - query_token)
        scores = torch.einsum(
            "hd,khd->hk",
            query[query_token].float(),
            key_dequant[:query_seq_len],
        )
        scores.mul_(head_size**-0.5)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probs, value_dequant[:query_seq_len]))
    return torch.stack(outputs).half()


def benchmark_case(
    case: dict[str, object],
    warmup: int,
    repeats: int,
    iterations: int,
    reference_max_context: int,
    backend: str,
) -> Result:
    for _ in range(warmup):
        run_attention(case, backend)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run_attention(case, backend)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)

    max_abs_error = None
    mean_abs_error = None
    context_tokens = int(case["context_tokens"])
    if context_tokens <= reference_max_context:
        actual = run_attention(case, backend).float()
        expected = reference_attention(case).float()
        error = (actual - expected).abs()
        max_abs_error = float(error.max())
        mean_abs_error = float(error.mean())

    median_ms = statistics.median(samples)
    return Result(
        context_tokens=context_tokens,
        median_ms=median_ms,
        min_ms=min(samples),
        max_ms=max(samples),
        tokens_per_second_scanned=context_tokens / (median_ms / 1000.0),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        raise SystemExit(f"This benchmark targets SM75, found {capability}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    contexts = [int(value) for value in args.contexts.split(",") if value]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "torch": torch.__version__,
                "backend": args.backend,
                "pid": os.getpid(),
                "shape": {
                    "query_heads": args.query_heads,
                    "kv_heads": args.kv_heads,
                    "head_size": args.head_size,
                    "block_size": args.block_size,
                    "query_tokens": args.query_tokens,
                    "permute_blocks": args.permute_blocks,
                },
            },
            sort_keys=True,
        )
    )

    for context_tokens in contexts:
        case = make_case(
            context_tokens,
            args.query_tokens,
            args.query_heads,
            args.kv_heads,
            args.head_size,
            args.block_size,
            args.permute_blocks,
            device,
        )
        if args.backend == "sm75-cuda":
            from vllm.v1.attention.ops.sm75_int8_decode_attention import workspace

            case["workspaces"] = workspace(context_tokens, device)
        elif args.backend == "sm75-integrated":
            from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

            case["impl"] = TritonAttentionImpl(
                num_heads=args.query_heads,
                head_size=args.head_size,
                scale=args.head_size**-0.5,
                num_kv_heads=args.kv_heads,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="int8_per_token_head",
            )
            case["metadata"] = SimpleNamespace(
                max_query_len=args.query_tokens,
                seq_lens_cpu=torch.tensor([context_tokens], dtype=torch.int32),
                block_table=case["block_table"],
                mm_prefix_range_tensor=None,
                is_for_cudagraph_capture=False,
            )
        result = benchmark_case(
            case,
            args.warmup,
            args.repeats,
            args.iterations,
            args.reference_max_context,
            args.backend,
        )
        print(json.dumps(asdict(result), sort_keys=True))
        del case
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
