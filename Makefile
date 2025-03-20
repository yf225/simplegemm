NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a
NVCC_LDFLAGS = -lcublas -lcuda # --keep # -lineinfo

gemm: main.cu gemm.cu pingpong.cu stmatrix.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $< -o $@

maxreg: maxreg.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $^ -o $@

# CLANG_FLAGS = -x cuda -std=c++17 -O3 -DNDEBUG -w \
# 	--cuda-path=/usr/local/cuda-12.8 \
# 	--cuda-gpu-arch=sm_90a \
# 	-fPIE -Wno-psabi -fno-strict-aliasing

# CLANG_LDFLAGS = -L/usr/local/cuda-12.8/lib64 -L/usr/local/cuda-12.8/lib -lcublas -lcudart -lcuda

# gemm: main.cu gemm.cu pingpong.cu stmatrix.cu
# 	clang++ $(CLANG_FLAGS) $(CLANG_LDFLAGS) $< -o $@

# maxreg: maxreg.cu
# 	clang++ $(CLANG_FLAGS) $(CLANG_LDFLAGS $< -o $@


# nvcc -std=c++17 -O3 -DNDEBUG -w --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a -lcublas -lcuda  main.cu -o gemm

# clang++ -x cuda -std=c++17 -O3 -DNDEBUG -w   --cuda-path=/usr/local/cuda-12.8   --cuda-gpu-arch=sm_90a   main.cu -o gemm   -fPIE -Wno-psabi -fno-strict-aliasing   -L/usr/local/cuda-12.8/lib64 -L/usr/local/cuda-12.8/lib   -lcublas -lcudart -lcuda
