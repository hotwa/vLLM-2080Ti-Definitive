#!/usr/bin/env bash
set -eu

MODE=${1:-baseline}
: "${MODEL_PATH:?Set MODEL_PATH to the Ornith FP8 MTP checkpoint}"

VLLM_EXECUTABLE=${VLLM_EXECUTABLE:-vllm}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18086}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-ornith-35b-vllm-lab}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-80000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-52}
MTP_K=${MTP_K:-3}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_xml}

case "$MODE" in
  baseline)
    unset VLLM_SM75_INT8_DECODE
    ;;
  sm75)
    export VLLM_SM75_INT8_DECODE=1
    export VLLM_SM75_INT8_DECODE_MIN_TOKENS_PER_QUERY=${VLLM_SM75_INT8_DECODE_MIN_TOKENS_PER_QUERY:-256}
    ;;
  *)
    printf 'usage: %s baseline|sm75\n' "$0" >&2
    exit 2
    ;;
esac

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.5}
export MAX_JOBS=${MAX_JOBS:-4}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_INT8KV_FA_PREFILL=${VLLM_INT8KV_FA_PREFILL:-1}
export VLLM_INT8KV_FA_CONTINUATION_DEQUANT=${VLLM_INT8KV_FA_CONTINUATION_DEQUANT:-1}
export VLLM_INT8KV_FA_CASCADE_DEQUANT=${VLLM_INT8KV_FA_CASCADE_DEQUANT:-1}
export VLLM_INT8KV_FA_CASCADE_TILE_TOKENS=${VLLM_INT8KV_FA_CASCADE_TILE_TOKENS:-65536}

exec "$VLLM_EXECUTABLE" serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --dtype half \
  --quantization fp8 \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
  --kv-cache-dtype int8_per_token_head \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K}}" \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --mamba-cache-mode align \
  --gdn-prefill-backend flashqla_legacy \
  --language-model-only \
  --skip-mm-profiling \
  --enforce-eager
