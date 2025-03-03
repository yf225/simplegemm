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

__global__ void testFill(bf16* X, int M, int N, int parity) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int m_idx = idx % M;
  int n_idx = idx / M;
  if (m_idx >= M || n_idx >= N)
    return;
  if (parity < 0) {
    X[idx] = (m_idx == n_idx) ? 1.0 : 0.0;
  } else {
    X[idx] = idx;
  }

  // int v = (idx % 8 - 4);
  // //v = (v >= 0) ? v + 1 : v;
  // //X[idx] = (bf16)(v * parity);
  // X[idx] = (float)(clock() % 8) / 8.0 - 0.5;
}

cublasHandle_t cublas_handle;
void runCublasGemmBF16(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
  float alpha = 1, beta = 0;
  // C(column major) = A(row major) * B(column major)
  cublasStatus_t status = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_16BF,
    K, B, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "CUBLAS error: %d\n", status);
    exit(EXIT_FAILURE);
  }
}

__global__ void gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    int m_idx = idx % M;
    int n_idx = idx / M;
    float sum = 0.0;
    for (int k = 0; k < K; k++) {
      sum += __bfloat162float(A[m_idx * K + k]) * __bfloat162float(B[k + n_idx * K]);
    }
    C[m_idx + n_idx * M]= __float2bfloat16(sum);
  }
}

int main() {
  int m = 6 * 11 * 128;
  int n = 6 * 12 * 128;
  int k = 256;

  m = k = 8;
  n = 16;

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
  testFill<<<cdiv(numel, 1024), 1024>>>(A, m, k, 1);
  testFill<<<cdiv(numel, 1024), 1024>>>(B, n, k, -1);
  check(cudaGetLastError());

  // Generate cuBLAS reference.
  cublasCreate(&cublas_handle);
  runCublasGemmBF16(m, n, k, A, B, Cref);

  // Run test kernel.
  gemm<<<cdiv(m * n, 1024), 1024>>>(A, B, C, m, n, k);

  // Print a slab of matrix for sanity.
  bf16* hostA = (bf16*)malloc(sizeof(bf16) * numel);
  bf16* hostB = (bf16*)malloc(sizeof(bf16) * numel);
  check(cudaMemcpy(hostA, A, sizeof(bf16) * m * k, cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hostB, B, sizeof(bf16) * n * k, cudaMemcpyDeviceToHost));

  for (int i = 0; i< 8; i++) {
      for (int j = 0; j < 8; j++) {
        printf("  %6.2f", __bfloat162float(hostA[i * k + j]));
      }
      printf("\n");
  }
  printf("\n");
  for (int i = 0; i< 8; i++) {
      for (int j = 0; j < 8; j++) {
        printf("  %6.2f", __bfloat162float(hostB[i + j * k]));
      }
      printf("\n");
  }
  printf("\n");

  bf16* hostM = (bf16*)malloc(sizeof(bf16) * numel);
  auto print = [&] (bf16* X) {
    check(cudaMemcpy(hostM, X, sizeof(bf16) * numel, cudaMemcpyDeviceToHost));
    check(cudaDeviceSynchronize());
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        printf("  %6.2f", __bfloat162float(hostM[i + j * m]));
      }
      printf("\n");
    }
    printf("\n");
  };
  //print(A);
  //print(B);
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
