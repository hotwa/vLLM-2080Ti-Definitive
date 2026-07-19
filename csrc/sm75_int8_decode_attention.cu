// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

using namespace nvcuda;

constexpr int kQueryHeads = 8;
constexpr int kKvHeads = 1;
constexpr int kHeadSize = 256;
constexpr int kPartTokens = 32;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kPartThreads = kWarpSize * kWarpsPerBlock;
constexpr int kMaxQueryTokens = 4;

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__global__ void sm75_int8_decode_part_kernel(
    const half* __restrict__ query,
    const int8_t* __restrict__ key,
    const int8_t* __restrict__ value,
    const float* __restrict__ key_scale,
    const float* __restrict__ value_scale,
    const int32_t* __restrict__ block_table,
    float* __restrict__ partial_output,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    int seq_len,
    int num_query_tokens,
    int num_parts,
    int block_size,
    float softmax_scale,
    int64_t query_stride_token,
    int64_t query_stride_head,
    int64_t key_stride_block,
    int64_t key_stride_slot,
    int64_t key_stride_head,
    int64_t key_stride_dim,
    int64_t value_stride_block,
    int64_t value_stride_slot,
    int64_t value_stride_head,
    int64_t value_stride_dim,
    int64_t key_scale_stride_block,
    int64_t key_scale_stride_slot,
    int64_t key_scale_stride_head,
    int64_t value_scale_stride_block,
    int64_t value_scale_stride_slot,
    int64_t value_scale_stride_head,
    int64_t block_table_stride) {
#if __CUDA_ARCH__ >= 750
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int part = blockIdx.x;
  const int token_start = part * kPartTokens;

  __shared__ __align__(16)
      int8_t query_quant[kMaxQueryTokens][kQueryHeads][kHeadSize];
  __shared__ __align__(16) int8_t key_tile[kPartTokens][kHeadSize];
  __shared__ __align__(16) half value_tile[kPartTokens][kHeadSize];
  __shared__ __align__(16)
      int32_t score_int[kMaxQueryTokens][kQueryHeads][kPartTokens];
  __shared__ __align__(16)
      half probability[kMaxQueryTokens * kQueryHeads][kPartTokens];
  __shared__ float query_scale_shared[kMaxQueryTokens][kQueryHeads];
  __shared__ float key_scale_shared[kPartTokens];
  __shared__ float value_scale_shared[kPartTokens];
  __shared__ int32_t physical_block_shared[kPartTokens];

  for (int query_token = warp; query_token < num_query_tokens;
       query_token += kWarpsPerBlock) {
#pragma unroll
    for (int query_head = 0; query_head < kQueryHeads; ++query_head) {
      float local_absmax = 0.0f;
#pragma unroll
      for (int dim = lane; dim < kHeadSize; dim += kWarpSize) {
        const float query_value = __half2float(
            query[query_token * query_stride_token +
                  query_head * query_stride_head + dim]);
        local_absmax = fmaxf(local_absmax, fabsf(query_value));
      }
      const float absmax = warp_reduce_max(local_absmax);
      const float quant_scale = fmaxf(absmax / 127.0f, 1.0e-8f);
      if (lane == 0) {
        query_scale_shared[query_token][query_head] = quant_scale;
      }
#pragma unroll
      for (int dim = lane; dim < kHeadSize; dim += kWarpSize) {
        const float query_value = __half2float(
            query[query_token * query_stride_token +
                  query_head * query_stride_head + dim]);
        const int rounded = __float2int_rn(query_value / quant_scale);
        query_quant[query_token][query_head][dim] =
            static_cast<int8_t>(max(-127, min(127, rounded)));
      }
    }
  }
  __syncthreads();

  const int global_token = token_start + lane;
  if (warp == 0) {
    if (global_token < seq_len) {
      const int logical_block = global_token / block_size;
      const int slot = global_token % block_size;
      const int32_t physical_block =
          block_table[logical_block * block_table_stride];
      physical_block_shared[lane] = physical_block;
      key_scale_shared[lane] = key_scale[
          physical_block * key_scale_stride_block +
          slot * key_scale_stride_slot + 0 * key_scale_stride_head];
      value_scale_shared[lane] = value_scale[
          physical_block * value_scale_stride_block +
          slot * value_scale_stride_slot + 0 * value_scale_stride_head];
    } else {
      physical_block_shared[lane] = 0;
      key_scale_shared[lane] = 1.0f;
      value_scale_shared[lane] = 1.0f;
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < kPartTokens * kHeadSize;
       index += kPartThreads) {
    const int token = index / kHeadSize;
    const int dim = index % kHeadSize;
    const int token_index = token_start + token;
    if (token_index < seq_len) {
      const int slot = token_index % block_size;
      const int32_t physical_block = physical_block_shared[token];
      key_tile[token][dim] = key[
          physical_block * key_stride_block + slot * key_stride_slot +
          0 * key_stride_head + dim * key_stride_dim];
      const int8_t value_quant = value[
          physical_block * value_stride_block + slot * value_stride_slot +
          0 * value_stride_head + dim * value_stride_dim];
      value_tile[token][dim] = __float2half_rn(
          static_cast<float>(value_quant) * value_scale_shared[token]);
    } else {
      key_tile[token][dim] = 0;
      value_tile[token][dim] = __float2half_rn(0.0f);
    }
  }
  __syncthreads();

  for (int query_token = warp; query_token < num_query_tokens;
       query_token += kWarpsPerBlock) {
    wmma::fragment<wmma::accumulator, 8, 32, 16, int> accumulator;
    wmma::fill_fragment(accumulator, 0);
#pragma unroll
    for (int offset_k = 0; offset_k < kHeadSize; offset_k += 16) {
      wmma::fragment<
          wmma::matrix_a,
          8,
          32,
          16,
          signed char,
          wmma::row_major>
          query_fragment;
      wmma::fragment<
          wmma::matrix_b,
          8,
          32,
          16,
          signed char,
          wmma::col_major>
          key_fragment;
      wmma::load_matrix_sync(
          query_fragment,
          reinterpret_cast<const signed char*>(
              &query_quant[query_token][0][offset_k]),
          kHeadSize);
      wmma::load_matrix_sync(
          key_fragment,
          reinterpret_cast<const signed char*>(&key_tile[0][offset_k]),
          kHeadSize);
      wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
    }
    wmma::store_matrix_sync(
        &score_int[query_token][0][0],
        accumulator,
        kPartTokens,
        wmma::mem_row_major);
  }
  __syncthreads();

  for (int index = threadIdx.x;
       index < kMaxQueryTokens * kQueryHeads * kPartTokens;
       index += kPartThreads) {
    reinterpret_cast<half*>(probability)[index] = __float2half_rn(0.0f);
  }
  __syncthreads();

  for (int query_token = warp; query_token < num_query_tokens;
       query_token += kWarpsPerBlock) {
    const int query_seq_len =
        seq_len - (num_query_tokens - 1 - query_token);
#pragma unroll
    for (int query_head = 0; query_head < kQueryHeads; ++query_head) {
      float score = -INFINITY;
      if (global_token < query_seq_len) {
        score = static_cast<float>(score_int[query_token][query_head][lane]) *
            query_scale_shared[query_token][query_head] *
            key_scale_shared[lane] * softmax_scale;
      }
      const float max_score = warp_reduce_max(score);
      const float probability_value = global_token < query_seq_len
          ? __expf(score - max_score)
          : 0.0f;
      const float sum_probability = warp_reduce_sum(probability_value);
      probability[query_token * kQueryHeads + query_head][lane] =
          __float2half_rn(probability_value);
      if (lane == 0) {
        const int stats_index =
            (part * kMaxQueryTokens + query_token) * kQueryHeads + query_head;
        partial_max[stats_index] = max_score;
        partial_sum[stats_index] = sum_probability;
      }
    }
  }
  __syncthreads();

  const int output_rows = num_query_tokens * kQueryHeads;
  const int row_start = (warp / 2) * 16;
  const int first_column = (warp % 2) * 16;
  if (row_start < output_rows) {
#pragma unroll
    for (int column_start = first_column; column_start < kHeadSize;
         column_start += 32) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
      for (int token_start = 0; token_start < kPartTokens;
           token_start += 16) {
        wmma::fragment<
            wmma::matrix_a,
            16,
            16,
            16,
            half,
            wmma::row_major>
            probability_fragment;
        wmma::fragment<
            wmma::matrix_b,
            16,
            16,
            16,
            half,
            wmma::row_major>
            value_fragment;
        wmma::load_matrix_sync(
            probability_fragment,
            &probability[row_start][token_start],
            kPartTokens);
        wmma::load_matrix_sync(
            value_fragment,
            &value_tile[token_start][column_start],
            kHeadSize);
        wmma::mma_sync(
            accumulator,
            probability_fragment,
            value_fragment,
            accumulator);
      }
      float* output_tile = partial_output +
          static_cast<int64_t>(part) * kMaxQueryTokens * kQueryHeads *
              kHeadSize +
          row_start * kHeadSize + column_start;
      wmma::store_matrix_sync(
          output_tile, accumulator, kHeadSize, wmma::mem_row_major);
    }
  }
#endif
}

__global__ void sm75_int8_decode_reduce_kernel(
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    half* __restrict__ output,
    int num_parts,
    int64_t output_stride_token,
    int64_t output_stride_head) {
  extern __shared__ float shared[];
  float* part_alpha = shared;
  float* reduction = shared + num_parts;

  const int query_head = blockIdx.x;
  const int query_token = blockIdx.y;
  const int dim = threadIdx.x;

  float local_max = -INFINITY;
  for (int part = dim; part < num_parts; part += blockDim.x) {
    local_max = fmaxf(
        local_max,
        partial_max[
            (part * kMaxQueryTokens + query_token) * kQueryHeads +
            query_head]);
  }
  reduction[dim] = local_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (dim < offset) {
      reduction[dim] = fmaxf(reduction[dim], reduction[dim + offset]);
    }
    __syncthreads();
  }
  const float global_max = reduction[0];

  float local_sum = 0.0f;
  for (int part = dim; part < num_parts; part += blockDim.x) {
    const float alpha = __expf(
        partial_max[
            (part * kMaxQueryTokens + query_token) * kQueryHeads +
            query_head] -
        global_max);
    part_alpha[part] = alpha;
    local_sum += partial_sum[
                     (part * kMaxQueryTokens + query_token) * kQueryHeads +
                     query_head] *
        alpha;
  }
  reduction[dim] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (dim < offset) {
      reduction[dim] += reduction[dim + offset];
    }
    __syncthreads();
  }
  const float denominator = reduction[0];
  __syncthreads();

  float output_value = 0.0f;
  for (int part = 0; part < num_parts; ++part) {
    const int64_t partial_index =
        ((static_cast<int64_t>(part) * kMaxQueryTokens + query_token) *
             kQueryHeads + query_head) *
            kHeadSize +
        dim;
    output_value += partial_output[partial_index] * part_alpha[part];
  }
  output[query_token * output_stride_token +
         query_head * output_stride_head + dim] =
      __float2half_rn(output_value / denominator);
}

void check_common_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& key_scale,
    const torch::Tensor& value_scale,
    const torch::Tensor& block_table,
    const torch::Tensor& output) {
  TORCH_CHECK(
      query.is_cuda() && key.is_cuda() && value.is_cuda() &&
          key_scale.is_cuda() && value_scale.is_cuda() &&
          block_table.is_cuda() && output.is_cuda(),
      "all tensors must be CUDA tensors");
  const int device = query.get_device();
  TORCH_CHECK(
      key.get_device() == device && value.get_device() == device &&
          key_scale.get_device() == device && value_scale.get_device() == device &&
          block_table.get_device() == device && output.get_device() == device,
      "all tensors must be on the same CUDA device");
  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(
      properties->major == 7 && properties->minor == 5,
      "SM75 INT8 decode attention requires compute capability 7.5");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "query must be float16");
  TORCH_CHECK(key.scalar_type() == torch::kInt8, "key must be int8");
  TORCH_CHECK(value.scalar_type() == torch::kInt8, "value must be int8");
  TORCH_CHECK(
      key_scale.scalar_type() == torch::kFloat32 &&
          value_scale.scalar_type() == torch::kFloat32,
      "KV scales must be float32");
  TORCH_CHECK(
      block_table.scalar_type() == torch::kInt32,
      "block_table must be int32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");
  TORCH_CHECK(
      query.dim() == 3 && query.size(0) >= 1 &&
          query.size(0) <= kMaxQueryTokens &&
          query.size(1) == kQueryHeads && query.size(2) == kHeadSize &&
          query.stride(2) == 1,
      "query must be [1..4, 8, 256]");
  TORCH_CHECK(
      key.dim() == 4 && value.dim() == 4 && key.size(1) > 0 &&
          value.size(1) == key.size(1) && key.size(2) == kKvHeads &&
          value.size(2) == kKvHeads && key.size(3) >= kHeadSize &&
          value.size(3) >= kHeadSize,
      "KV cache must be [num_blocks, block_size, 1, >=256]");
  TORCH_CHECK(
      key_scale.dim() == 3 && value_scale.dim() == 3 &&
          key_scale.size(0) == key.size(0) &&
          value_scale.size(0) == value.size(0) &&
          key_scale.size(1) == key.size(1) &&
          value_scale.size(1) == value.size(1) &&
          key_scale.size(2) == kKvHeads &&
          value_scale.size(2) == kKvHeads,
      "KV scales must be [num_blocks, block_size, 1]");
  TORCH_CHECK(
      block_table.dim() == 2 && block_table.size(0) >= 1,
      "block_table must be [num_sequences, max_blocks]");
  TORCH_CHECK(
      output.sizes() == query.sizes() && output.stride(2) == 1,
      "output shape must match query and have contiguous head dimension");
}

}  // namespace

void sm75_int8_decode_attention_out(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    torch::Tensor block_table,
    torch::Tensor partial_output,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor output,
    int64_t seq_len,
    double softmax_scale) {
  check_common_inputs(
      query, key, value, key_scale, value_scale, block_table, output);
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");
  TORCH_CHECK(
      seq_len >= query.size(0), "seq_len must cover every query token");
  TORCH_CHECK(
      seq_len <= block_table.size(1) * key.size(1),
      "seq_len exceeds block_table capacity");
  const int num_query_tokens = static_cast<int>(query.size(0));
  const int num_parts = static_cast<int>((seq_len + kPartTokens - 1) / kPartTokens);
  TORCH_CHECK(
      partial_output.scalar_type() == torch::kFloat32 &&
          partial_output.dim() == 4 &&
          partial_output.size(0) >= num_parts &&
          partial_output.size(1) == kMaxQueryTokens &&
          partial_output.size(2) == kQueryHeads &&
          partial_output.size(3) == kHeadSize && partial_output.is_contiguous() &&
          partial_output.get_device() == query.get_device(),
      "partial_output must be contiguous [>=num_parts, 4, 8, 256] "
      "float32");
  TORCH_CHECK(
      partial_max.scalar_type() == torch::kFloat32 &&
          partial_sum.scalar_type() == torch::kFloat32 &&
          partial_max.dim() == 3 && partial_sum.dim() == 3 &&
          partial_max.size(0) >= num_parts &&
          partial_sum.size(0) >= num_parts &&
          partial_max.size(1) == kMaxQueryTokens &&
          partial_sum.size(1) == kMaxQueryTokens &&
          partial_max.size(2) == kQueryHeads &&
          partial_sum.size(2) == kQueryHeads && partial_max.is_contiguous() &&
          partial_sum.is_contiguous() &&
          partial_max.get_device() == query.get_device() &&
          partial_sum.get_device() == query.get_device(),
      "partial stats must be contiguous [>=num_parts, 4, 8] float32");

  auto stream = c10::cuda::getCurrentCUDAStream(query.get_device());
  sm75_int8_decode_part_kernel<<<num_parts, kPartThreads, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      key.data_ptr<int8_t>(),
      value.data_ptr<int8_t>(),
      key_scale.data_ptr<float>(),
      value_scale.data_ptr<float>(),
      block_table.data_ptr<int32_t>(),
      partial_output.data_ptr<float>(),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      static_cast<int>(seq_len),
      num_query_tokens,
      num_parts,
      static_cast<int>(key.size(1)),
      static_cast<float>(softmax_scale),
      query.stride(0),
      query.stride(1),
      key.stride(0),
      key.stride(1),
      key.stride(2),
      key.stride(3),
      value.stride(0),
      value.stride(1),
      value.stride(2),
      value.stride(3),
      key_scale.stride(0),
      key_scale.stride(1),
      key_scale.stride(2),
      value_scale.stride(0),
      value_scale.stride(1),
      value_scale.stride(2),
      block_table.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t shared_bytes =
      static_cast<size_t>(num_parts + kHeadSize) * sizeof(float);
  sm75_int8_decode_reduce_kernel<<<
      dim3(kQueryHeads, num_query_tokens),
      kHeadSize,
      shared_bytes,
      stream>>>(
      partial_output.data_ptr<float>(),
      partial_max.data_ptr<float>(),
      partial_sum.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      num_parts,
      output.stride(0),
      output.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "attention_out",
      &sm75_int8_decode_attention_out,
      "SM75 INT8 paged decode/verification attention (out variant)");
}
