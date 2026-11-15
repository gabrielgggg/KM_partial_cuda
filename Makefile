# Build the solver (one binary, both back ends, both precisions) with nvcc from the NVIDIA HPC SDK.
SDK   ?= /opt/nvidia/hpc_sdk/Linux_x86_64/26.3
CUDA  ?= $(SDK)/cuda/13.1
ARCH  ?= sm_75
NVCC   = $(CUDA)/bin/nvcc
FLAGS  = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler "-O3 -march=native -fopenmp" -Xptxas -O3
PY    ?= $(HOME)/venvs/torch/bin/python

bin/km: src/km.cu src/model.h src/shared.h src/cpu_backend.h src/gpu_backend.cuh src/post.h
	mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ src/km.cu -lgomp

clean:
	rm -rf bin
