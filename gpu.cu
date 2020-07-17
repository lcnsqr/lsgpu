#include <stdio.h>
#include "gpu.h"

// Cuda cores per multiprocessor from Compute Capability
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
    { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
    { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
    { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
    { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
    { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
    { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
    { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
    { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
    { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
    { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class
		{ 0x72, 64 }, // Xavier Generation (SM 7.2) GV10B class
		{ 0x75, 64 }, // Turing Generation (SM 7.5) TU102 class
		{ 0x80, 64 }, // Ampere Generation (SM 8.0) GA10x class
    {   -1, -1 }
  };
  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }
  // If we don't find the values, we default use the previous one to run properly
  //fprintf(stderr, "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
  return nGpuArchCoresPerSM[index-1].Cores;
}

// How many local GPUs
int gpu_count(){
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess){
    //fprintf(stderr, "Error: cudaGetDeviceCount returns %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    deviceCount = -1;
  }
  return deviceCount;
}

// Load GPU properties
int gpu_properties(cudaDeviceProp* deviceProp, int deviceCount){
  cudaError_t error_id;
  for (int c = 0; c < deviceCount; c++){
    error_id = cudaGetDeviceProperties(&deviceProp[c], c);
    if (error_id != cudaSuccess) return -1;
  }
  return deviceCount;
}

// Descriptive GPU details
int gpu_description(char* buffer){

  int deviceCount = gpu_count();
  if ( deviceCount < 0 ) return -1;

  cudaDeviceProp *deviceProp = (cudaDeviceProp *)malloc(sizeof(cudaDeviceProp) * deviceCount);
  gpu_properties(deviceProp, deviceCount);
  
	int driverVersion = 0;
	int runtimeVersion = 0;

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  // GPU chipset family
  char family[32];

  // Description entry
  char buf_entry[4096];

  for (int c = 0; c < deviceCount && c < MAX_GPU; c++){

    // Get GPU family
    switch ( deviceProp[c].major ){
      case 3: sprintf(family, "Kepler"); break;
      case 5: sprintf(family, "Maxwell"); break;
      case 6: sprintf(family, "Pascal"); break;
      case 7:
        if ( deviceProp[c].minor == 0 )
          sprintf(family, "Volta");
        if ( deviceProp[c].minor == 2 )
          sprintf(family, "Xavier");
        if ( deviceProp[c].minor == 5 )
          sprintf(family, "Turing");
      break;
      case 8: sprintf(family, "Ampere"); break;
    }

    snprintf(buf_entry, 4096, "\
  Device:          %d\n\
  Name:            %s\n\
  Family:          %s\n\
  Capability:      %d.%d\n\
  Cores / MP:      %d\n\
  Global Memory:   %.0f MB\n\
  Driver:          %d.%d\n\
  Runtime:         %d.%d\n", 
    c,
    deviceProp[c].name,
    family,
    deviceProp[c].major, deviceProp[c].minor,
    _ConvertSMVer2Cores(deviceProp[c].major, deviceProp[c].minor),
    (float)deviceProp[c].totalGlobalMem/1048576.0f,
    driverVersion/1000, (driverVersion%100)/10,
    runtimeVersion/1000, (runtimeVersion%100)/10);

    // Add to description
    strncat(buffer, buf_entry, 4096);
  }

  free(deviceProp);

  return deviceCount;
}
