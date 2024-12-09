
#define advectSinglePassCS advectSinglePassCS_densityNoEmit_SST

#define ENABLE_VELOCITY 0
#define ENABLE_EMIT 0

#define ENABLE_SST 1
#define ENABLE_VTR 0

#include "advectSinglePassCS.hlsl"