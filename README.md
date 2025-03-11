# Pingpong GEMM from scratch

I wrote this kernel to see if I could match CUTLASS's ["pingpong" GEMM
algorithm](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#hopper-warp-specialization)
using hand-written CUDA.  I used https://github.com/pranjalssh/fast.cu by Pranjal Shankhdhar as a
starting point, having been heavily inspired by the fantastic blog post
[Outperforming cuBLAS on
H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).

You can run a quick check of the kernel with:
```
make gemm && ./gemm
```

And run a sweep through a bunch of different shapes with:
```
python setup.py develop && python benchmark.py
```
