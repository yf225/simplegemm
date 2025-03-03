#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdio.h>

using bf16 = __nv_bfloat16;

void checkCudaErrors(cudaError_t error, const char* file, int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define check(err) checkCudaErrors(err, __FILE__, __LINE__)

int cdiv(int m, int n) {
  return (m + n - 1) / n;
}

__global__ void testFill(bf16* X, int N, int parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int v = (idx % 8 - 4);
  v = (v >= 0) ? v + 1 : v;
  X[idx] = (bf16)(v * parity);
}

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    N, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error: %d\n", status);
    exit(EXIT_FAILURE);
  }
}

int main() {
  int m = 6 * 11 * 128;
  int n = 6 * 12 * 128;
  int k = 1280;

  int max = 16384;

  // Allocate matrices
  __nv_bfloat16* A;
  __nv_bfloat16* B;
  __nv_bfloat16* C;
  __nv_bfloat16* Cref;

  check(cudaMalloc((void**)&A, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&B, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&C, sizeof(bf16) * max * max));
  check(cudaMalloc((void**)&Cref, sizeof(bf16) * max * max));

  // Fill with test data.
  int numel = max * max;
  testFill<<<numel / 1024, 1024>>>(A, numel, 1);
  testFill<<<numel / 1024, 1024>>>(B, numel, -1);
  check(cudaGetLastError());

  // Generate cuBLAS reference.
  cublasCreate(&cublas_handle);
  runCublasGemmBF16(m, n, k, A, B, C);

  // Run test kernel.

  // Print a slab of matrix for sanity.
  bf16* hostM = (bf16*)malloc(sizeof(bf16) * numel);
  auto print = [&] (bf16* X) {
    check(cudaMemcpy(hostM, X, sizeof(bf16) * numel, cudaMemcpyDeviceToHost));
    check(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        printf("  %6.2f", __bfloat162float(hostM[i * max + j]));
      }
      printf("\n");
    }
    printf("\n");
  };
  print(A);
  print(B);
  print(C);
  print(Cref);

  // Test against cuBLAS reference.
  bf16* hostC = nullptr;
  bf16* hostCref = nullptr;
  if (false) {
    hostC = (bf16*)malloc(sizeof(bf16) * m * n);
    hostCref = (bf16*)malloc(sizeof(bf16) * m * n);

    check(cudaMemcpy(hostC, C, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hostCref, Cref, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));

    exit(0);
    for (int i = 0; i < m * n; i++) {
      float cv = __bfloat162float(hostC[i]);
      float crefv = __bfloat162float(hostCref[i]);
      if (std::abs(cv - crefv) > 1e-5) {
        fprintf(stderr, "Failed tolerance check at idx %d, C=%f, Cref=%f\n", i, cv, crefv);
      }
    }
  }

  // Benchmark test kernel.

  // Free resources.
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(Cref);
  free(hostM);
  free(hostC);
  free(hostCref);
  return 0;
}
