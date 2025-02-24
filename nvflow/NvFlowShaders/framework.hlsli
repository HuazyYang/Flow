#ifndef HEREAFTER_FRAMEWORK_HLSLI
#define HEREAFTER_FRAMEWORK_HLSLI

/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define NV_FLOW_SHADER_UTILS 1

#define NvFlowUint uint
#define NvFlowUint2 uint2
#define NvFlowUint3 uint3
#define NvFlowUint4 uint4

#define NvFlowInt2 int2
#define NvFlowInt3 int3
#define NvFlowInt4 int4

#define NvFlowFloat2 float2
#define NvFlowFloat3 float3
#define NvFlowFloat4 float4

#define NvFlowFloat4x4 float4x4

#include "../NvFlow/NvFlowShader.h"

SamplerState borderSampler : register(s0);
SamplerState borderPointSampler : register(s1);
SamplerState wrapSampler : register(s2);
SamplerState wrapPointSampler : register(s3);
SamplerState clampSampler : register(s4);
SamplerState clampPointSampler : register(s5);

int3 tableVal_to_coord(uint val) {
  uint valInv = ~val;
  return int3((valInv >> 0) & 0x3FF, (valInv >> 10) & 0x3FF,
              (valInv >> 20) & 0x3FF);
}

uint coord_to_tableVal(int3 coord) {
  uint valInv = (coord.x << 0) | (coord.y << 10) | (coord.z << 20);
  return ~valInv;
}

#define DISPATCH_ID_TO_VIRTUAL(blockListSRV, params)                           \
  int3 DispatchIDToVirtual(uint3 tidx) {                                       \
    uint blockID = tidx.x >> params.blockDimBits.x;                            \
    int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);                 \
    int3 vidx = (vBlockIdx << params.blockDimBits.xyz) |                       \
                (tidx & (params.blockDim.xyz - int3(1, 1, 1)));                \
    return vidx;                                                               \
  }

#define DISPATCH_ID_TO_VIRTUAL_LINEAR(blockListSRV, params)                    \
  int3 DispatchIDToVirtual(uint3 tidx, out int3 vBlockIdx,                     \
                           out uint3 cellIdx) {                                \
    uint blockID = tidx.y;                                                     \
    cellIdx =                                                                  \
        int3(tidx.x % params.linearBlockDim.x,                                 \
             (tidx.x / params.linearBlockDim.x) % params.linearBlockDim.y,     \
             tidx.x / (params.linearBlockDim.x * params.linearBlockDim.y));    \
    int3 vidx = int3(-1, -1, -1);                                              \
    vBlockIdx = int3(-1, -1, -1);                                              \
    if (cellIdx.z < params.linearBlockDim.z) {                                 \
      vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);                    \
      vidx = (vBlockIdx << params.blockDimBits.xyz) +                          \
             (cellIdx - params.linearBlockOffset.xyz);                         \
    }                                                                          \
    return vidx;                                                               \
  }

#define DISPATCH_ID_TO_VIRTUAL_1D(blockListSRV, params)                        \
  int3 DispatchIDToVirtual(uint3 tidx) {                                       \
    uint blockID = tidx.x >> params.blockDimBits.w;                            \
    int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);                 \
    int3 cellIdx;                                                              \
    cellIdx.x = tidx.x;                                                        \
    cellIdx.y = cellIdx.x >> params.blockDimBits.x;                            \
    cellIdx.z = cellIdx.y >> params.blockDimBits.y;                            \
    cellIdx = cellIdx & (params.blockDim.xyz - int3(1, 1, 1));                 \
    int3 vidx = (vBlockIdx << params.blockDimBits.xyz) | cellIdx;              \
    return vidx;                                                               \
  }

//! Use this version when each thread operates on a 2x2x2 cell set
#define DISPATCH_ID_TO_VIRTUAL2(blockListSRV, params)                          \
  int3 DispatchIDToVirtual(uint3 tidx) {                                       \
    uint blockID = tidx.x >> (params.blockDimBits.x - 1);                      \
    int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);                 \
    int3 vidx = (vBlockIdx << params.blockDimBits.xyz) |                       \
                ((tidx << 1) & (params.blockDim.xyz - int3(1, 1, 1)));         \
    return vidx;                                                               \
  }

#endif /* HEREAFTER_FRAMEWORK_HLSLI */
