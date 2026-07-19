# Experimental SM75 INT8 decode attention

This branch adds an opt-in CUDA decode path for Turing SM75 GPUs. It targets
the TP=2 rank shape used by Ornith/Qwen3.5 35B: 8 query heads, one KV head,
head size 256, FP16 queries, and `int8_per_token_head` paged KV cache.

The kernel dynamically quantizes each FP16 query head to signed INT8 and
computes QK with the SM75 `m8n32k16` signed-char WMMA operation and INT32
accumulation. Four warps share each K/V tile: one warp handles each MTP query
for QK, while all four cooperate on the FP16 Tensor Core PV matrix multiply.
Segmented softmax and the cross-part reduction accumulate in FP32. It handles
one decode token or up to four causal MTP verification tokens and supports
arbitrary runtime KV block sizes and paged block tables.

This is a deliberately narrow prototype. It is disabled by default, only
supports batch size one, falls back for unsupported attention features, and
does not run during CUDA graph capture. Full-model testing therefore uses
`--enforce-eager`.

## Microbenchmark

Run the existing Triton path and the integrated SM75 path with identical data:

```bash
python benchmarks/kernels/benchmark_sm75_int8_attention.py \
  --backend triton --query-tokens 4 --reference-max-context 80000

VLLM_SM75_INT8_DECODE=1 \
python benchmarks/kernels/benchmark_sm75_int8_attention.py \
  --backend sm75-integrated --query-tokens 4 \
  --reference-max-context 80000
```

Measured on two RTX 2080 Ti 22 GB cards, one TP rank at a time:

| Context | Triton | SM75 CUDA | Speedup | Maximum absolute error |
|---:|---:|---:|---:|---:|
| 5,000 | 3.559 ms | 0.169 ms | 21.06x | 0.000900 |
| 38,000 | 18.991 ms | 1.012 ms | 18.77x | 0.000298 |
| 80,000 | 40.109 ms | 2.080 ms | 19.28x | 0.000183 |

These final measurements use the model's runtime block size of 2,112 and a
random physical page order. The integration retains a conservative
256-token-per-query crossover threshold so unsupported or very small calls
continue through the existing path.

## Full-model run

The launcher expects the FP8 Ornith MTP checkpoint and a vLLM executable from
this fork's environment:

```bash
MODEL_PATH=/path/to/ornith-35b-fp8-e4m3-mtp \
VLLM_EXECUTABLE=/path/to/.venv/bin/vllm \
benchmarks/kernels/run_sm75_int8_full_model.sh baseline

MODEL_PATH=/path/to/ornith-35b-fp8-e4m3-mtp \
VLLM_EXECUTABLE=/path/to/.venv/bin/vllm \
benchmarks/kernels/run_sm75_int8_full_model.sh sm75
```

The tested configuration uses TP=2, MTP3, an 80K maximum model length,
`int8_per_token_head` KV, and 52 GPU blocks. The override leaves enough memory
for FlashInfer and GDN prefill workspaces on the 22 GB cards.

| Input / output | Existing vLLM decode | SM75 decode | Speedup |
|---:|---:|---:|---:|
| 5K / 128 | 54.82 tok/s | 80.19 tok/s | 1.46x |
| 38K / 128 | 11.37 tok/s | 78.93 tok/s | 6.94x |
| 79K / 128 | 4.17 tok/s | 44.56 tok/s | 10.69x |

These are single-request serving measurements with fixed random inputs,
temperature 0.01, ignored EOS, and identical seeds. A second 38K run after the
OpenCode stress test measured 78.68 tok/s, a 0.3% decode change. Enabling
`VLLM_MARLIN_USE_ATOMIC_ADD=1` regressed the same test to 75.70 tok/s and is
not recommended for this setup.

## OpenCode CLI stress result

OpenCode 1.17.13 was configured with the local OpenAI-compatible endpoint,
`--enable-auto-tool-choice`, and `--tool-call-parser qwen3_xml`. One session
was grown through ten rounds to approximately 60K input tokens by attaching
the kernel, integration, and benchmark sources. The server completed 13 chat
requests and three benchmark requests with HTTP 200, no OOM, CUDA error, Xid,
or GPU-memory growth. A 60K `glob -> read -> final answer` chain succeeded, but
took 108 seconds because every agent step re-prefilled the full conversation.

The qualitative result is mixed: most 34K-60K source-analysis answers and the
cross-round sentinel recall were accurate, but one 14K answer was garbled and
one 60K round made an unnecessary tool call. This implementation is therefore
a research result, not a production default. Longer soak testing, fixed agent
quality suites, concurrency, CUDA-graph-safe sequence lengths, and broader
model shapes remain unverified.
