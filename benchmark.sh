#!/bin/bash

export TORCHINDUCTOR_CACHE_DIR=/tmp/pingpong_matmul_experiments_20250305_9998_2
export TORCHINDUCTOR_CUTLASS_DIR=/data/users/bertrand/cutlass
export TORCHINDUCTOR_CUTLASS_ALLOWLIST='128x128x64_1x1x1.*pingpong_epi_tma'
export TORCHINDUCTOR_CUTLASS_DENYLIST='stream_k'
export TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL=0201
#export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export USE_IR_LOC=ttgir

DATE=$(date +%s)
export TRITON_DUMP_DIR=$(realpath "dump.$DATE")
#export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=4}"

if hg root; then
    RUN_COMMAND="buck2 run @fbcode//mode/opt -c fbcode.platform010_cuda_version=12.4 -c fbcode.nvcc_arch=h100a :11-pingpong-matmul"
else
    RUN_COMMAND="python benchmark.py"
fi

#if [ -z "$TRITON_OVERRIDE_DIR" ]; then
if false; then
    export TRITON_OVERRIDE_DIR=$(realpath "override.$DATE")

    echo $TRITON_DUMP_DIR
    echo $TRITON_OVERRIDE_DIR

    rm -rf $TRITON_DUMP_DIR $TRITON_OVERRIDE_DIR

    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 ./denoise-h100.sh $RUN_COMMAND
    cp -r $TRITON_DUMP_DIR $TRITON_OVERRIDE_DIR
    TTGIR_PATH=$(find $TRITON_OVERRIDE_DIR -name 'matmul_persistent_tma_ws_pingpong_kernel.ttgir')
    find $TRITON_OVERRIDE_DIR -type f -delete
    cp matmul_persistent_tma_ws_pingpong_kernel.ttgir $TTGIR_PATH
fi

export BENCHMARK_CUTLASS=1
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 ./denoise-h100.sh $RUN_COMMAND
