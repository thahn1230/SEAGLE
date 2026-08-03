import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"   # force: shell may carry 12.0
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ["PATH"]
from torch.utils.cpp_extension import load
HERE = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="seagle_int4_ext",
    sources=[os.path.join(HERE, "torch_binding.cpp"),
             os.path.join(HERE, "w4a4_sm89.cu"),
             os.path.join(HERE, "w4a4_fused.cu")],
    extra_include_paths=[
        HERE,
        "/data/thahn1230/quarot/third-party/cutlass/include",
        "/data/thahn1230/quarot/third-party/cutlass/tools/util/include",
    ],
    extra_cuda_cflags=["-O3", "-std=c++17",
                       "-gencode=arch=compute_89,code=sm_89"],
    verbose=False)
if __name__ == "__main__":
    print("ext ok:", ext.__file__ if hasattr(ext, "__file__") else ext)
