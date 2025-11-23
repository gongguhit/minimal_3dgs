#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                    "-Xcompiler", "-fno-stack-protector",
                    "-Xcompiler", "-fno-stack-check",
                    "-Xcompiler", "-D_FORTIFY_SOURCE=0"
                ],
                "cxx": [
                    "-fno-stack-protector",
                    "-fno-stack-check",
                    "-D_FORTIFY_SOURCE=0",
                    "-D_GLIBCXX_USE_CXX11_ABI=0"
                ]
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
