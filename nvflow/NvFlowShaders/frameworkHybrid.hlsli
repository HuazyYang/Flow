#ifndef HEREAFTER_FRAMEWORKHYBRID_HLSLI
#define HEREAFTER_FRAMEWORKHYBRID_HLSLI

/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "framework.hlsli"

// Note: assumes CB has NvFlowUint4 isVTR

#ifndef ENABLE_VTR
#define ENABLE_VTR 1
#endif
#ifndef ENABLE_SST
#define ENABLE_SST 1
#endif

#if (SIMULATE_REDUNDANT)

#if (ENABLE_VTR && ENABLE_SST)

#define DISPATCH_ID_TO_VIRTUAL_AND_REAL_RIDX(blockTableSRV, params)            \
  if (isVTR.x != 0u) {                                                         \
    ridx = vidx;                                                               \
  } else {                                                                     \
    int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);              \
    ridx = rBlockIdx * params.linearBlockDim.xyz + cellIdx;                    \
  }

#elif (ENABLE_SST)

#define DISPATCH_ID_TO_VIRTUAL_AND_REAL_RIDX(blockTableSRV, params)            \
  int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);                \
  ridx = rBlockIdx * params.linearBlockDim.xyz + cellIdx;

#elif (ENABLE_VTR)

#define DISPATCH_ID_TO_VIRTUAL_AND_REAL_RIDX(blockTableSRV, params) ridx = vidx;

#endif

#define DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, blockTableSRV, params)   \
  DISPATCH_ID_TO_VIRTUAL_LINEAR(blockListSRV, params)                          \
  void DispatchIDToVirtualAndReal(uint3 tidx, out int3 vidx, out int3 ridx) {  \
    int3 vBlockIdx;                                                            \
    uint3 cellIdx;                                                             \
    vidx = DispatchIDToVirtual(tidx, vBlockIdx, cellIdx);                      \
    DISPATCH_ID_TO_VIRTUAL_AND_REAL_RIDX(blockTableSRV, params)                \
  }

#endif

#if (SIMULATE_ONCE)

#define DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, blockTableSRV, params)   \
  VIRTUAL_TO_REAL(VirtualToRealDispatch, blockTableSRV, params)                \
  DISPATCH_ID_TO_VIRTUAL(blockListSRV, params)                                 \
  void DispatchIDToVirtualAndReal(uint3 tidx, out int3 vidx, out int3 ridx) {  \
    vidx = DispatchIDToVirtual(tidx);                                          \
    ridx = VirtualToRealDispatch(vidx);                                        \
  }

#endif

// ********************* Dynamic VTR/SST ************************
#if (ENABLE_VTR && ENABLE_SST)

#define VIRTUAL_TO_REAL_LINEAR(name, blockTableSRV, params)                    \
  float3 name(float3 vidx) {                                                   \
    if (params.isVTR.x != 0) {                                                 \
      return vidx;                                                             \
    } else {                                                                   \
      float3 vBlockIdxf = params.blockDimInv.xyz * vidx;                       \
      int3 vBlockIdx = int3(floor(vBlockIdxf));                                \
      int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);            \
      float3 rBlockIdxf = float3(rBlockIdx);                                   \
      float3 ridx =                                                            \
          float3(params.linearBlockDim.xyz * rBlockIdx) +                      \
          float3(params.blockDim.xyz) * (vBlockIdxf - float3(vBlockIdx)) +     \
          float3(params.linearBlockOffset.xyz);                                \
      return ridx;                                                             \
    }                                                                          \
  }

#define VIRTUAL_TO_REAL_LINEAR2(name, params)                                  \
  float3 name(float3 vidx, int3 rBlockIdx, int3 vBlockIdx) {                   \
    if (params.isVTR.x != 0) {                                                 \
      return vidx;                                                             \
    } else {                                                                   \
      float3 vBlockIdxf = params.blockDimInv.xyz * vidx;                       \
      float3 rBlockIdxf = float3(rBlockIdx);                                   \
      float3 ridx =                                                            \
          float3(params.linearBlockDim.xyz * rBlockIdx) +                      \
          float3(params.blockDim.xyz) * (vBlockIdxf - float3(vBlockIdx)) +     \
          float3(params.linearBlockOffset.xyz);                                \
      return ridx;                                                             \
    }                                                                          \
  }

#define VIRTUAL_TO_REAL(name, blockTableSRV, params)                           \
  int3 name(int3 vidx) {                                                       \
    if (params.isVTR.x != 0) {                                                 \
      return vidx;                                                             \
    } else {                                                                   \
      int3 vBlockIdx = vidx >> params.blockDimBits.xyz;                        \
      int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);            \
      int3 ridx = (rBlockIdx << params.blockDimBits.xyz) |                     \
                  (vidx & (params.blockDim.xyz - int3(1, 1, 1)));              \
      return ridx;                                                             \
    }                                                                          \
  }

// ********************* VTR Only *******************************
#elif (ENABLE_VTR)

#define VIRTUAL_TO_REAL_LINEAR(name, blockTableSRV, params)                    \
  float3 name(float3 vidx) { return vidx; }

#define VIRTUAL_TO_REAL_LINEAR2(name, params)                                  \
  float3 name(float3 vidx, int3 rBlockIdx, int3 vBlockIdx) { return vidx; }

#define VIRTUAL_TO_REAL(name, blockTableSRV, params)                           \
  int3 name(int3 vidx) { return vidx; }

// ******************** SST Only ********************************
#elif (ENABLE_SST)

#define VIRTUAL_TO_REAL_LINEAR(name, blockTableSRV, params)                    \
  float3 name(float3 vidx) {                                                   \
    float3 vBlockIdxf = params.blockDimInv.xyz * vidx;                         \
    int3 vBlockIdx = int3(floor(vBlockIdxf));                                  \
    int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);              \
    float3 rBlockIdxf = float3(rBlockIdx);                                     \
    float3 ridx =                                                              \
        float3(params.linearBlockDim.xyz * rBlockIdx) +                        \
        float3(params.blockDim.xyz) * (vBlockIdxf - float3(vBlockIdx)) +       \
        float3(params.linearBlockOffset.xyz);                                  \
    return ridx;                                                               \
  }

#define VIRTUAL_TO_REAL_LINEAR2(name, params)                                  \
  float3 name(float3 vidx, int3 rBlockIdx, int3 vBlockIdx) {                   \
    float3 vBlockIdxf = params.blockDimInv.xyz * vidx;                         \
    float3 rBlockIdxf = float3(rBlockIdx);                                     \
    float3 ridx =                                                              \
        float3(params.linearBlockDim.xyz * rBlockIdx) +                        \
        float3(params.blockDim.xyz) * (vBlockIdxf - float3(vBlockIdx)) +       \
        float3(params.linearBlockOffset.xyz);                                  \
    return ridx;                                                               \
  }

#define VIRTUAL_TO_REAL(name, blockTableSRV, params)                           \
  int3 name(int3 vidx) {                                                       \
    int3 vBlockIdx = vidx >> params.blockDimBits.xyz;                          \
    int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);              \
    int3 ridx = (rBlockIdx << params.blockDimBits.xyz) |                       \
                (vidx & (params.blockDim.xyz - int3(1, 1, 1)));                \
    return ridx;                                                               \
  }

#endif

#define SAMPLE_LINEAR_3D(name, type, dataSRV, blockTableSRV, params)           \
  VIRTUAL_TO_REAL_LINEAR(VirtualToReal##name, blockTableSRV, params);          \
  type name(float3 vidx) {                                                     \
    float3 ridx = VirtualToReal##name(vidx);                                   \
    type value =                                                               \
        dataSRV.SampleLevel(borderSampler, params.dimInv.xyz * ridx, 0);       \
    return value;                                                              \
  }

#define SAMPLE_LINEAR_3D_NORM(name, type, dataSRV, blockTableSRV, params)      \
  VIRTUAL_TO_REAL_LINEAR(VirtualToReal##name, blockTableSRV, params);          \
  type name(float3 vidxNorm) {                                                 \
    float3 ridx = VirtualToReal##name(params.vdim.xyz * vidxNorm);             \
    type value =                                                               \
        dataSRV.SampleLevel(borderSampler, params.dimInv.xyz * ridx, 0);       \
    return value;                                                              \
  }

#define SAMPLE_POINT_3D(name, type, dataSRV, blockTableSRV, params)            \
  VIRTUAL_TO_REAL(VirtualToReal##name, blockTableSRV, params);                 \
  type name(int3 vidx) {                                                       \
    int3 ridx = VirtualToReal##name(vidx);                                     \
    type value = dataSRV[ridx];                                                \
    return value;                                                              \
  }

#define OUTPUT_3D(name, type, dataUAV, blockTableSRV, params)                  \
  VIRTUAL_TO_REAL(VirtualToReal##name, blockTableSRV, params);                 \
  void name(int3 vidx, type value) {                                           \
    int3 ridx = VirtualToReal##name(vidx);                                     \
    dataUAV[ridx] = value;                                                     \
  }

#endif /* HEREAFTER_FRAMEWORKHYBRID_HLSLI */
