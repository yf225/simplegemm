#include <cuda.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>

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

__host__ static inline CUtensorMap create_tma_desc(bf16* gmem, int M, int N, int BLOCK_M, int BLOCK_N)  {
  CUtensorMap tma_desc;
  // TODO: Check these requirements against the HW spec.
  assert(BLOCK_N >= 64);
  assert(N % 64 == 0);

  // TODO: cdiv?
  // TODO" why the 64 inner dim?
  uint64_t shape[] = {64, M, N / 64};
  uint64_t stride[] = {sizeof(bf16) * N, 64 * sizeof(bf16)};
  uint32_t box_shape[] = {64, BLOCK_M, BLOCK_N / 64};
  uint32_t box_stride[] = {1, 1, 1};

  auto result = cuTensorMapEncodeTiled(
            &tma_desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            gmem,
            shape,
            stride,
            box_shape,
            box_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "TMA desc creation failed\n");
    exit(EXIT_FAILURE);
  }

  return tma_desc;
}

template <uint32_t REGS>
__device__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

template <uint32_t REGS>
__device__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(REGS));
}

__device__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;\n"
      :: "r"(bar_ptr), "r"(thread_count + transaction_count));
}

__device__ void wait_barrier(uint64_t* bar, int phase) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni DONE;\n"
      "bra.uni LAB_WAIT;\n"
      "DONE:\n"
      "}\n"
      :: "r"(mbar_ptr), "r"(phase)
  );
}

__device__ void arrive_barrier(uint64_t* bar, int count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(bar_ptr), "r"(count) : "memory");
}

__device__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(bar_ptr), "r"(bytes));
}

__device__ void tma_load(bf16* dst, void const* const src_tma_desc, uint64_t* bar, int n, int m) {
  uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_desc);
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      ::
       "r"(dst_ptr),
       "l"(tma_ptr),
       "r"(bar_ptr),
       "n"(0),
       "r"(m),
       "r"(n / 64)
      : "memory");
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

__global__ void naive_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
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

void run_naive_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  naive_gemm<<<cdiv(M * N, 1024), 1024>>>(A, B, C, M, N, K);
}

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 64;
constexpr int NUM_SMS = 132;
constexpr int STAGES = 3;
constexpr int WARPGROUP_SIZE = 128;
constexpr int WARPGROUPS = 3;
constexpr int NUM_THREADS = WARPGROUPS * WARPGROUP_SIZE;

struct SharedStorage {
  alignas(128) bf16 A[BLOCK_M * BLOCK_K * STAGES];
  alignas(128) bf16 B[BLOCK_K * BLOCK_N * STAGES];
};

__global__ __launch_bounds__(NUM_THREADS) void gemm(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, bf16* C, int M, int N, int K) {
  // Producer buffers for A and B.
  extern __shared__ __align__(128) uint8_t dynamic_smem[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(dynamic_smem);

  // Barriers.
  __shared__ __align__(8) uint64_t prod[STAGES];
  __shared__ __align__(8) uint64_t cons[STAGES];

  int tid = threadIdx.x;
  int wgid = tid / WARPGROUP_SIZE;
  int wg_tid = tid % WARPGROUP_SIZE;

  // Init barriers.
  if (tid == 0) {
    for (int i = 0; i < STAGES; i++) {
      init_barrier(&prod[i], 0, 1);
      init_barrier(&cons[i], 0, WARPGROUPS - 1);
    }
  }
  __syncthreads();

  if (wgid == 0) {
    // Producer warpgroup.
    setmaxnreg_dec<40>();
    int phase = 0;
    // Mainloop.
    int m = 0, n = 0;
    if (wg_tid == 0) {
      for (int k = 0; k < K; k += BLOCK_K) {
        // Wait for consumer.
        // TODO: stage and phase update.
        wait_barrier(&cons[0], phase);
        // Set expect bytes for TMA.
        expect_bytes(&prod[0], sizeof(bf16) * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N));
        // Load A.
        // TODO: use proper stage
        tma_load(&smem.A[0], &A, &prod[0], k * BLOCK_K, m * BLOCK_M);
        // Load B.
        tma_load(&smem.B[0], &B, &prod[0], k * BLOCK_K, n * BLOCK_N);
      }
    }
  } else {
    // Consumer warpgroup.
    setmaxnreg_inc<240>();
    int phase = 0;
    if (wg_tid == 0) {
      arrive_barrier(&cons[0], 1);
    }
    // Mainloop.
    for (int k = 0; k < K; k += BLOCK_K) {
      // Wait for producer.
      wait_barrier(&prod[0], phase);
      // Perform wgmma.
      // Arrive at consumer.
    }
    // Write back to gmem.
  }
  // __syncthreads();
  // if (tid == 128) {
  //   printf("smem.A:\n");
  //   for (int i = 0; i < BLOCK_M; i++) {
  //     for (int j = 0; j < BLOCK_K; j++) {
  //       printf("  %6.2f", __bfloat162float(smem.A[i * BLOCK_K + j]));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  //   printf("smem.B:\n");
  //   for (int i = 0; i < BLOCK_K; i++) {
  //     for (int j = 0; j < BLOCK_N; j++) {
  //       printf("  %6.2f", __bfloat162float(smem.B[i + j * BLOCK_K]));
  //     }
  //     printf("\n");
  //   }
  //   printf("\n");
  // }
}

void run_gemm(bf16* A, bf16* B, bf16* C, int M, int N, int K) {
  // Compute necessary shared memory for buffers.
  size_t smem_size = sizeof(SharedStorage);
  check(cudaFuncSetAttribute(gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Set up TMA descriptors
  auto descA = create_tma_desc(A, M, K, BLOCK_M, BLOCK_K);
  auto descB = create_tma_desc(B, N, K, BLOCK_N, BLOCK_K);

  // Launch kernel!
  gemm<<<1, NUM_THREADS, smem_size>>>(descA, descB, C, M, N, K);
  check(cudaDeviceSynchronize());
  check(cudaGetLastError());
}

int main() {
  // int m = 6 * 11 * 128;
  // int n = 6 * 12 * 128;
  // int k = 512;

  //m = k = 8;
  //n = 16;

  int m = 128;
  int n = 256;
  int k = 64;

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
  run_gemm(A, B, C, m, n, k);

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
  if (true) {
    hostC = (bf16*)malloc(sizeof(bf16) * m * n);
    hostCref = (bf16*)malloc(sizeof(bf16) * m * n);

    check(cudaMemcpy(hostC, C, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(hostCref, Cref, sizeof(bf16) * m * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < m * n; i++) {
      float cv = __bfloat162float(hostC[i]);
      float crefv = __bfloat162float(hostCref[i]);
      if (std::abs(cv - crefv) > 1e-5) {
        fprintf(stderr, "Failed tolerance check at idx %d, C=%f, Cref=%f\n", i, cv, crefv);
        exit(EXIT_FAILURE);
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
