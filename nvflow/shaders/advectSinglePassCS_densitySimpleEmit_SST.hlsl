
#define advectSinglePassCS advectSinglePassCS_densitySimpleEmit_SST

#define ENABLE_VELOCITY 0
#define ENABLE_EMIT 1
#define ENABLE_SIMPLE_EMIT 1

#define ENABLE_SST 1
#define ENABLE_VTR 0

#include "advectSinglePassCS.hlsl"