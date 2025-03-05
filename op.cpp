// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ATen/ATen.h" // @manual
#include "torch/extension.h" // @manual

void run_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K);

at::Tensor gemm(at::Tensor a, at::Tensor b) {
  // a (m x k), b (k x n)
  auto c = a.new_empty({b.size(1), a.size(0)}).transpose();
  run_gemm(
      a.data_ptr(),
      b.data_ptr(),
      c.data_ptr(),
      a.size(0),
      b.size(1),
      a.size(1));
  return c;
}

TORCH_LIBRARY(gemm, m) {
  m.def("gemm", &gemm);
}
