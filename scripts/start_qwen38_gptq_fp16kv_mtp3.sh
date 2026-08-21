#!/bin/bash
set -euo pipefail

/home/lingyuzeng/project/vllm-2080ti/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model /home/lingyuzeng/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
  --served-model-name qwen38-gptq-fp16kv-128K-mtp3-text-only \
  --dtype half \
  --tensor-parallel-size 2 \
  --generation-config vllm \
  --max-model-len 128000 \
  --enable-chunked-prefill \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --language-model-only \
  --skip-mm-profiling \
  --disable-log-stats \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[4],"max_cudagraph_capture_size":4}' \
  2>&1 | tee /home/lingyuzeng/project/vllm-2080ti/run-logs/vllm-qwen38-gptq-fp16kv-mtp3-$(date +%Y%m%d-%H%M%S).log