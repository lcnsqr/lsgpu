#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "gpu.h"

// CPU

// Process rank
int rank;
// Number of processing units
int size;
// Root process
int root;

int main(int argc, char **argv){

  // MPI setup
  MPI_Status status;
  MPI_Init(&argc, &argv);

  // How many parallel processes are running?
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the rank of the current process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // The last process is the root
  root = size - 1;

  int deviceCount = gpu_count();

  char **buffer = (char**)malloc(sizeof(char*)*deviceCount);
  for (int c = 0; c < deviceCount; c++){
    buffer[c] = (char*)malloc(sizeof(char)*4096);
  }
  gpu_description(buffer);

  if ( rank == root ){
    for (int c = 0; c < deviceCount; c++){
      printf("%s\n", buffer[c]);
    }
  }

  MPI_Finalize();

	exit(EXIT_SUCCESS);
}
