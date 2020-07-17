#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include "gpu.h"

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

  if ( argc < 2 ){
    if ( rank == root ){
      fprintf(stderr, "Output filename not provided.\n");
    }
    MPI_Finalize();
    exit(0);
  }

  char *filename = argv[1];

  // Description entry
  char buf_entry[4096];
  // Buffer to store local descriptions of local devices
  char *buf_local = (char*)malloc(sizeof(char)*(1+MAX_GPU)*4096);
  memset(buf_local, 0, sizeof(char)*(1+MAX_GPU)*4096);

  // Add node name and device count to the description
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(hostname, &len);

  // Device count
  int deviceCount = gpu_count();

  // Format node details
  snprintf(buf_entry, 4096, "\
Node:            %s\n\
Devices:         %d\n", 
  hostname, 
  deviceCount);

  // Add to description
  strncat(buf_local, buf_entry, 4096);

  // Read GPU details
  gpu_description(buf_local);

  // Global buffer
  char *buf = NULL;
  if ( rank == root ){
    buf = (char*)malloc(size*sizeof(char)*(1+MAX_GPU)*4096);
  }

  // Gather descriptions from all processes
  MPI_Gather(buf_local, (1+MAX_GPU)*4096, MPI_CHAR, buf, (1+MAX_GPU)*4096, MPI_CHAR, root, MPI_COMM_WORLD);

  if ( rank == root ){
    // Save descriptions
    FILE *file = fopen(filename, "w");
    for (int r = 0; r < size; r++){
      fprintf(file, "%s\n", &buf[r * (1+MAX_GPU)*4096]);
    }
    fclose(file);
    free(buf);
  }

  free(buf_local);

  MPI_Finalize();

	exit(EXIT_SUCCESS);
}
