# KVarN k4v2 SM75 Feasibility Note

Status: on hold. This is an implementation possibility, not a recommended
profile route.

## Summary

KVarN `kvarn_k4v2_g128` showed strong KV-cache capacity/admission behavior on
dual RTX 2080 Ti, but the current SM75 runtime does not qualify as a performance
route. It should not be promoted unless a real fast-attention path is proven for
Qwen3.6 27B's GQA layout.

## Evidence

Test environment:

- Model: Qwen3.6 27B FP8 family checkpoint
- GPUs: dual RTX 2080 Ti, TP=2
- KV cache dtype: `kvarn_k4v2_g128`
- Standard short throughput probe: `pp4096 / tg128`

Observed short-probe throughput:

| Route | Prefill tok/s | Decode tok/s | Result |
|---|---:|---:|---|
| SDPA fallback after FA2 gating | 783.48 | 7.32 | Not viable |
| Fused verify forced for continuation | 865.81 | 7.38 | Not viable |

The capacity side was promising: the 64K service admitted roughly 500K KV tokens
under the tested settings. However, the short-probe decode speed is far below
the existing FP8/INT4 production routes, so longer context tests are not useful
until the short path is fixed.

## Fast-Path Boundary

The current KVarN attention backend is wired around `flash_attn_varlen` for the
fast varlen path. On SM75, FA2 is not available, so the backend falls back to
SDPA or KVarN fused verify paths.

FlashInfer was probed as a possible replacement, but the installed FlashInfer
path does not directly cover the target Qwen3.6 27B shape on SM75:

- Equal-head small probes can run.
- Qwen3.6 27B uses GQA: `num_attention_heads=24`,
  `num_key_value_heads=4`, `head_dim=256`.
- FlashInfer `single_prefill_with_kv_cache` and
  `BatchPrefillWithRaggedKVCacheWrapper` both failed on this GQA shape with
  CUDA invalid-argument errors on RTX 2080 Ti.

FlashQLA is not a direct substitute here. The existing FlashQLA work accelerates
Qwen GDN / linear-attention prefill, while KVarN's problematic continuation path
needs a full-attention varlen or paged fast path.

## Promotion Criteria

KVarN k4v2 should only be reconsidered if all of the following are true:

- The Qwen3.6 27B GQA shape runs on SM75 through FlashInfer, FlashQLA, or an
  equivalent full-attention fast path.
- `pp4096 / tg128` short-probe throughput is competitive with existing
  recommended FP8/INT4 routes.
- Output quality smoke passes before any long-context capacity testing.

Until then, KVarN k4v2 remains an on-hold capacity experiment rather than a
profile candidate.
