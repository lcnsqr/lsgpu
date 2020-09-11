# CUDA directory
CUDADIR=/usr/local/cuda

MPICC=mpicc

# Variables for MPI/CUDA linking
CUDA_LD_FLAGS = -L$(CUDADIR)/lib64 -lcudart
MPI_LD_FLAGS = $(shell mpic++ --showme:link)

.PHONY: all
all: lsgpu

# MPI + OpenMP + GPU

gpu.o: gpu.cu
	$(CUDADIR)/bin/nvcc -o $@ -c $^

lsgpu.o: lsgpu.c
	mpic++ -o $@ -c -fopenmp $^

lsgpu: gpu.o lsgpu.o
	g++ -o $@ gpu.o lsgpu.o $(CUDA_LD_FLAGS) $(MPI_LD_FLAGS) -lgomp

.PHONY: clean
clean:
	rm lsgpu gpu.o lsgpu.o
