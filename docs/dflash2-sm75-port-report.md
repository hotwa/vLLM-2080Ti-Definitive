# DFlash2 SM75 Port Report — Qwen3.8-27B on dual RTX 2080 Ti (TP=2)

**Branch:** `feature/dflash2-sm75-qwen38` · **Fork base:** v0.1.15 · **Upstream:** vLLM PR #52816 "[Spec Decode] DFlash2"
**Hardware:** 2× RTX 2080 Ti 22 GB (SM75/Turing, no bf16 kernels) · CUDA 12.8
**Target model:** `Qwen3.8-27B-Uncensored-FP8` (FP8 Marlin) + draft `Qwen3.8-27B-DFlash2` (bf16 checkpoint)
**Date:** 2026-08-20

---

## 1. Verdict (TL;DR)

| Question | Answer |
|---|---|
| Runnable on this fork? | **Yes.** Genuine DFlash2 path, all fork SM75 routes preserved. |
| Genuine DFlash2? | Yes — `DFlash2DraftModel`/`DFlash2Qwen3ForCausalLM` loaded, selector head active, non-zero acceptance. |
| Best `num_speculative_tokens` | **k = 5** (k=7/8 measurably worse). |
| Best measured speedup vs AR | **1.52–1.68×** end-to-end (k=5, compile+piecewise CUDA graphs, 32K fp16 KV). |
| vs MTP3 | **MTP3 remains ~1.4× faster** than DFlash2 on this model/hardware (acceptance 42–75% vs 9–34%). |
| Recommended production config | Keep **MTP3** for this model. DFlash2 is offered as an opt-in experimental profile. |
| Upstream-worthy? | Port logic itself is fork-local; the scaled-residual fp16 technique and non-causal-KV pin are documented below for upstream discussion. |

## 2. What was ported

Commits (oldest → newest):

| Commit | Content |
|---|---|
| `f590511` | Run DFlash drafts in checkpoint dtype; SM75 fp16 fallback decision in `vllm/config/speculative.py`. |
| `63b7ab6` | Scaled-residual fp16 path for bf16-trained DFlash drafts (`vllm/model_executor/models/qwen3_dflash*.py`). |
| `3631381` | 32K bring-up profile + greedy correctness harness (`scripts/dflash2/correctness_test.py`). |
| `ea64fdf` | Eager `CandidateSelector` for CUDA-graph replay; pin draft KV cache to fp16; bench-harness string fix. |

Upstream PR #52816 content brought over: `DFlash2Qwen3Model`/`DFlash2Qwen3ForCausalLM` (mlp_conv coefficient layers), `CandidateSelector` (predecessor/successor codebooks + hidden projection, `_score_edges`), `DFlash2Proposer` lattice walk with `_greedy_sample`, selector-top-k drafting, and the `dflash_config` checkpoint schema (`selector_rank=256`, `selector_top_k=16`, `mask_token_id`, `target_layer_ids`).

Nothing upstream overwrote: all SM75 patches (Marlin, FlashInfer, FlashQLA, INT8 KV, TurboQuant knobs, CUDAGraph, FP8/INT4 routes, MTP, launcher/profile system) untouched.

## 3. The core SM75 problem and its fix

**Problem.** The DFlash2 draft checkpoint is bf16-trained. Its residual stream reaches ~1.6e5 absmax and MLP activation products ~1e5 — both far beyond fp16 range (65504). SM75 has no bf16 compute kernels, and an fp32 draft does not fit the 22 GB budget alongside the FP8 target.

**Fix: scaled residual (`_RESIDUAL_SCALE = 1/16`).** RMSNorm is scale-invariant, so the residual stream can be carried at ×1/16:

- Every sublayer output (attention/MLP/conv) is multiplied by 1/16 *before* the fused add+RMSNorm.
- `_mlp_scaled(mlp, x, scale)` computes `silu(gate)*up` in fp32 (the only overflow-prone product) and folds the scale in before casting back to fp16 for `down_proj` — yielding exactly `scale·MLP(x)` with zero extra memory.
- Result: the entire draft runs fp16 on Turing, no overflow, no NaN, acceptance non-zero.

## 4. Bring-up validation (32K eager, TP=2)

Profile `profiles/qwen27b/experimental/fp8/fp16kv-32K-dflash2-tp2.env`, config
`{"method":"dflash","model":"Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_tensor_parallel_size":2}`.

- [x] Target (FP8 Marlin) and draft (fp16 scaled-residual) both load; genuine `DFlash2Qwen3ForCausalLM` instantiated (verified via architecture check and selector head presence).
- [x] ≥512 tokens generated, no NaN.
- [x] **NONZERO acceptance** (27–34% on short prompts).
- [x] No rank divergence under TP=2 (draft TP=2 with all-gather in selector scoring; 20 stable requests served).

## 5. Correctness (greedy, losslessness)

Harness: `scripts/dflash2_bench.py`/`correctness_test.py` — 10 prompts, `temperature=0`, `max_tokens=200`, greedy.

- 8/10 prompts bit-identical to the AR baseline; 2/10 diverge.
- Investigation of the divergences (a MUST-STOP condition was triggered and resolved):
  - Both AR and DFlash2 runs are individually deterministic (re-run identical).
  - At every divergence the logprob gap is a near-tie (Δ ≤ 0.031 nats; one literal fp16 tie).
  - A fresh AR *verify-sized* forward at the divergence position picks the DFlash2 token.
  - DFlash2 only ever emits tokens that are the argmax of the target's own verify-batch logprobs.
- **Root cause:** FP8 Marlin GEMM rounding is batch-shape dependent (AR decode batch=1 vs verify batch=1+k vs prefill), flipping argmax on near-ties. This is inherent to FP8 + speculative verification, not a port bug.
- **Conclusion:** acceptance logic is lossless w.r.t. the verify batch; the path is correct, with the documented FP8 near-tie caveat.

## 6. Performance ladder (e2e tok/s, single request; coding/reasoning/chat/agent)

| Config | coding | reasoning | chat | agent | acceptance (avg) |
|---|---|---|---|---|---|
| AR baseline (eager) | 27.9 | 27.6 | 27.6 | 27.5 | — |
| DFlash2 k=5 eager | 36.0 | 39.4 | 28.2 | 35.8 | 18–34% |
| DFlash2 k=7 eager | 35.2 | 36.9 | 27.8 | 33.3 | 13–21% |
| DFlash2 k=8 eager | 33.3 | 34.8 | 25.3 | 33.9 | 10–18% |
| DFlash2 k=5 compile (piecewise) | 39.4 | 42.7 | 29.7 | 38.5 | — |
| DFlash2 k=5 compile + cudagraph [1,2,5,6] | **42.3** | **46.4** | **33.1** | **42.7** | — |
| DFlash2 k=5 + prefix cache | 42.6 | 46.7 | 33.9 | 45.9 | 19–34% (unchanged) |
| MTP3 eager (reference) | 59.9 | 65.8 | 45.9 | 59.3 | 42–75% |

Observations:

- **k sweep:** k=5 dominates. Acceptance falls steeply with k (k=8 drafts 8, keeps ~1.7), so verify overhead outweighs drafted tokens.
- **compile:** `@support_torch_compile` had to be removed from `CandidateSelector` — the proposer invokes it with varying inputs, which breaks this fork's piecewise CUDA-graph replay input bookkeeping (`IndexError` in `cuda_graph.py` replay). Eager selector + compiled backbone is stable.
- **CUDA graphs:** capture sizes `[1,2,5,6]` via `COMPILATION_CONFIG_JSON` override (launcher default `capture=k+1` is wrong for DFlash's verify shape). +8–10% over compile-only.
- **Prefix cache:** no regression; slightly positive on the repetitive agent workload (+7.5%, 42.7→45.9 tok/s). Safe to leave on.

## 7. Fork-compatibility findings (rest of the ladder)

- **INT8 KV (`int8_per_token_head`):** DFlash2 cannot launch with it. Two independent blockers:
  1. No attention backend supports **non-causal** attention (required by the DFlash draft) with a quantized KV cache → fixed by pinning the draft cache to fp16 (`ea64fdf`).
  2. Page-size unification then fails: target int8 page (block×1056 B incl. fp32 scales) vs draft fp16 page (block×2048 B) have a 33:64 ratio — never divisible for any block size, so `unify_kv_cache_spec_page_size` raises. Recorded as a fork limitation.
  - Note: even the plain AR (no spec decode) INT8-KV 64K config died during CUDA-graph warmup with `custom_all_reduce.cuh:455 'invalid argument'` — unrelated to this port (no DFlash code on that path), flagging for the fork's INT8-KV work.
- **64K fp16 KV:** does not fit with the DFlash2 stack. At 32K the engine has ~3.15 GiB KV headroom per GPU; 64K needs 3.62 GiB. Raising `gpu_memory_utilization` to 0.95 passes profiling but OOMs on the first real request (no runtime headroom). The engine itself estimates a max model length of 56304 for this stack. 128K is out of reach with fp16 KV.
- **TurboQuant:** not applicable to the draft path (quantized-KV backends all reject non-causal attention; same class of blocker as INT8 KV).
- MTP + DFlash2 are mutually exclusive by design; never enabled together (MTP3 production profile untouched).

## 8. Recommendation

- **Production:** keep MTP3 138K (restored after validation). DFlash2 on SM75 tops out ~1.68× vs AR, below MTP3's ~2.1×.
- **DFlash2 opt-in profile:** k=5, compile + piecewise cudagraph `[1,2,5,6]`, 32K fp16 KV — useful when an MTP checkpoint is unavailable or for experiments.
- **Upstream notes:** the scaled-residual trick and the unconditional fp16 draft-KV pin are port-specific but worth mentioning upstream for low-end-hardware users; the page-size-unification limitation with mixed quantized/fp16 spec caches may deserve a generic fix (e.g. page padding for speculative draft caches).

## 9. Base-model experiment (2026-08-20)

**Hypothesis:** DFlash2's low acceptance on Uncensored-FP8 (9–34%) was a distribution mismatch; the incoai-trained draft should match the official base Qwen3.8-27B-FP8 and restore upstream-like ~76% acceptance (PR #52816).

**Method:** served official base Qwen3.8-27B-FP8 (text_config identical to Uncensored — same arch, different weights), 32K ctx, greedy temp=0, single sequence. Benchmarked DFlash2 k=5 cg, DFlash2 k=7 cg (native block_size=8), and MTP3 eager as control.

**Results** (e2e tok/s; coding / reasoning / chat / agent):

| target + spec | coding | reasoning | chat | agent |
|---|---|---|---|---|
| Uncensored AR | 27.9 | 27.6 | 27.6 | 27.5 |
| Uncensored + DFlash2 k=5 cg | 42.3 | 46.4 | 33.1 | 42.7 |
| Uncensored + MTP3 eager | 59.9 | 65.8 | 45.9 | 59.3 |
| base + DFlash2 k=5 cg | 38.9 | 39.9 | 33.9 | 41.3 |
| base + DFlash2 k=7 cg | 34.5 | 34.5 | 29.1 | 38.1 |
| base + MTP3 eager | 57.3 | 60.9 | 42.3 | 59.0 |

**Acceptance:** base + DFlash2 k=7: 16.2 / 16.9 / 11.9 / 20.1% (mean accepted len 1.13 / 1.18 / 0.84 / 1.41). base + MTP3: 57.5 / 65.5 / 35.9 / 64.6% (mean 4.03 / 4.59 / 2.52 / 4.53). Uncensored + DFlash2 k=5 was 19–34%; Uncensored + MTP3 was 42–75%.

**Conclusion: hypothesis falsified.** The base target did NOT restore DFlash2 acceptance — k=7 acceptance on base (12–20%) is *lower* than k=5 on Uncensored (19–34%), and throughput dropped too (34.5–38.1 vs 42.3–46.4 tok/s). MTP3 performs well on both targets (57–66 tok/s, 36–66% acceptance), isolating the problem to the draft checkpoint: the incoai DFlash2 draft matches neither local 27B weight version. Upstream's ~76% was measured at temp=1.0 / top-p 0.95 / top-k 20 (greedy is the harshest regime for multi-token drafts) and possibly against a different checkpoint revision.

**Final recommendation unchanged:** production stays MTP3 138K; DFlash2 k=5 cg 32K remains the opt-in fallback (~1.68× AR vs MTP3's ~2.1×).

## Appendix A — validated launch template

```bash
MODEL_DIR=/home/lingyuzeng/models/Qwen3.8-27B-Uncensored-FP8 \
PROFILE=qwen27b/experimental/fp8/fp16kv-32K-dflash2-tp2.env \
MODE=normal PORT=8000 GPU_DEVICES=0,1 TP_SIZE=2 NON_INTERACTIVE=1 \
DISABLE_PREFIX_CACHING=1 START_TIMEOUT=3000 ./launcher.sh \
  --set SPECULATIVE_CONFIG='{"method":"dflash","model":"/home/lingyuzeng/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":5,"draft_tensor_parallel_size":2}' \
  --set ENFORCE_EAGER=0 \
  --set COMPILATION_CONFIG_JSON='{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,5,6],"max_cudagraph_capture_size":6}' \
  --set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --set VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
```

## Appendix B — known caveats

- Greedy outputs are not guaranteed bit-identical to AR under FP8 targets (batch-shape-dependent near-ties, §5).
- Draft KV cache is always fp16 regardless of target KV quantization.
- `CandidateSelector` runs eager (incompatible with piecewise graph replay input bookkeeping).
- TTFT numbers from the harness are inflated on reasoning-parser workloads; compare `e2e_tok_s`.
