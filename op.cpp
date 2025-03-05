// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

void run_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K);

at::Tensor gemm(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({b.size(1), a.size(0)}).transpose(0, 1);
  run_gemm(
      a.data_ptr<bf16>(),
      b.data_ptr<bf16>(),
      c.data_ptr<bf16>(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

TORCH_LIBRARY(gemm, m) {
  m.def("gemm", &gemm);
}
