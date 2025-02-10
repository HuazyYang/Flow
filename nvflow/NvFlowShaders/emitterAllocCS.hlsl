/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#include "framework.hlsli"

#define EMITALLOC_DEFINES 1

#include "../NvFlow/emitterAllocShaderParams.h"

cbuffer params : register(b0)
{
	NvFlowUint4 emitterCount;
	NvFlowUint4 gridDim;
	NvFlowUint materialIdx;
	NvFlowUint matPad0;
	NvFlowUint matPad1;
	NvFlowUint matPad2;
};

RWTexture3D<uint> maskUAV : register(u0);

Buffer<float4> parameterSRV : register(t0);

groupshared NvFlowFloat4 sdata[EMITALLOC_DATA_SIZE];
groupshared NvFlowUint soffsets[512u];
groupshared NvFlowUint satomic;

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void emitterAllocCS(uint3 tidx : SV_DispatchThreadID, uint3 threadIdx : SV_GroupThreadID)
{
	uint3 blockIdx = tidx;
	uint threadIdx1D =
		(threadIdx.x) |
		(threadIdx.y << 3) |
		(threadIdx.z << 6);

	bool shouldAlloc = false;

	for (uint cullID = 0u; cullID < emitterCount.z; cullID++)
	{
		if (threadIdx1D == 0u)
		{
			satomic = 0u;
		}

		GroupMemoryBarrierWithGroupSync();

		uint emitterIdx = 512 * cullID + threadIdx1D;
		if (emitterIdx < emitterCount.x)
		{
			uint4 headerData = asuint(parameterSRV[emitterIdx]);
			uint3 boxmin = headerData.xyz & 0xFFFF;
			uint3 boxmax = headerData.xyz >> 16u;
			uint3 tmin = blockIdx & 0xFFFFFFF8;
			uint3 tmax = tmin + 8u;

			bool isOverlap = !(
				tmin.x > boxmax.x || boxmin.x > tmax.x ||
				tmin.y > boxmax.y || boxmin.y > tmax.y ||
				tmin.z > boxmax.z || boxmin.z > tmax.z
				);

			if (isOverlap)
			{
				uint allocIdx = 0u;
				InterlockedAdd(satomic, 1u, allocIdx);
				soffsets[allocIdx] = emitterIdx;
			}
		}

		GroupMemoryBarrierWithGroupSync();

		uint numOverlaps = satomic;
		for (uint idx = 0; idx < numOverlaps; idx++)
		{
			uint emitIdx = soffsets[idx];

			if (threadIdx1D < EMITALLOC_DATA_SIZE)
			{
				sdata[threadIdx1D] = parameterSRV[threadIdx1D + emitIdx * EMITALLOC_DATA_SIZE + emitterCount.y];
			}

			GroupMemoryBarrierWithGroupSync();

			if (all(blockIdx >= minVblockIdx.xyz) && all(blockIdx < maxVblockIdx.xyz))
			{
				float4 grid_ndc = 2.f * float4(vGridDimInv.xyz * (blockIdx + 0.5f.xxx), 1.f) - 1.f;
				float4 emitter_ndc1 = mul(grid_ndc, gridToEmitter1);
				float4 emitter_ndc2 = mul(grid_ndc, gridToEmitter2);

				bool materialMatch = (emitalloc_materialIdx == materialIdx) || (emitalloc_materialIdx == 0u);

				shouldAlloc = shouldAlloc || ((
					(all(emitter_ndc1.xyz >= -1.f.xxx) && all(emitter_ndc1.xyz <= +1.f.xxx)) ||
					(all(emitter_ndc2.xyz >= -1.f.xxx) && all(emitter_ndc2.xyz <= +1.f.xxx))
					) && materialMatch);
			}

			GroupMemoryBarrierWithGroupSync();
		}

		GroupMemoryBarrierWithGroupSync();
	}

	if (shouldAlloc)
	{
		maskUAV[blockIdx] = 1u;
	}
}