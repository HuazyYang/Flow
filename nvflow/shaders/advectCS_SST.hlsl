
#define advectCS advectCS_SST

#define THREAD_DIM_X 128
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

#define SIMULATE_REDUNDANT 1

#define ENABLE_SST 1
#define ENABLE_VTR 0

#include "advectCS.hlsl"