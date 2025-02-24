/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define THREAD_DIM_X 128
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

#include "../NvFlowShaders/framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowFloat4x4 positionToMask;
	NvFlowUint4 maskDim;
	NvFlowUint4 particleCount;
};

RWTexture3D<uint> maskUAV : register(u0);

Buffer<float4> positionSRV : register(t0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceAllocCS(uint3 tidx : SV_DispatchThreadID)
{
	float4 position = positionSRV[tidx.x];

	float4 gridPos = mul(position, positionToMask);

	float3 maskIdxf = maskDim.xyz * (0.5f * gridPos.xyz + 0.5f) - 0.5f;

	int3 maskIdx = floor(maskIdxf);

	maskUAV[maskIdx + int3(0, 0, 0)] = 1u;
	maskUAV[maskIdx + int3(1, 0, 0)] = 1u;
	maskUAV[maskIdx + int3(0, 1, 0)] = 1u;
	maskUAV[maskIdx + int3(1, 1, 0)] = 1u;
	maskUAV[maskIdx + int3(0, 0, 1)] = 1u;
	maskUAV[maskIdx + int3(1, 0, 1)] = 1u;
	maskUAV[maskIdx + int3(0, 1, 1)] = 1u;
	maskUAV[maskIdx + int3(1, 1, 1)] = 1u;
}