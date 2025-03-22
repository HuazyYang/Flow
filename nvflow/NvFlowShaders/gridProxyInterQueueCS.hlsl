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

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams srcParams;
	NvFlowShaderLinearParams params;
};

Buffer<uint> blockListSRV : register(t0);

Texture3D<float4> valueSRV : register(t1);
Texture3D<uint> blockTableSRV : register(t2);

RWTexture3D<float4> valueUAV : register(u0);

VIRTUAL_TO_REAL_LINEAR(VirtualToReal, blockTableSRV, srcParams)

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridProxyInterQueueCS(uint3 tidx : SV_DispatchThreadID)
{
	uint blockID = tidx.y;
	uint3 cellIdx = int3(
		tidx.x % params.linearBlockDim.x,
		(tidx.x / params.linearBlockDim.x) % params.linearBlockDim.y,
		tidx.x / (params.linearBlockDim.x * params.linearBlockDim.y));

	if (cellIdx.z < params.linearBlockDim.z)
	{
		int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);

		int3 srcVidx = vBlockIdx * params.blockDim.xyz + cellIdx - params.linearBlockOffset.xyz;
		int3 srcRidx = floor(VirtualToReal(float3(srcVidx)+0.5f.xxx));

		// lookup real block index
		int3 rBlockIdx = tableVal_to_coord(blockTableSRV[vBlockIdx]);
		int3 ridx = rBlockIdx * params.linearBlockDim.xyz + cellIdx;

		float4 value = valueSRV[srcRidx];

		valueUAV[ridx] = value;
	}
}