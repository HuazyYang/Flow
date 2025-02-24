/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "frameworkHybrid.hlsli"

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams velocityParams;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! Summary texture
RWTexture3D<float> summaryUAV : register(u0);

//! Velocity field input
Texture3D<float4> velocitySRV : register(t1);
Texture3D<uint> velocityBlockTable : register(t2);

SAMPLE_LINEAR_3D(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);

DISPATCH_ID_TO_VIRTUAL2(blockListSRV, velocityParams);

groupshared float sdata[2][8][8][8];

void sdata_write(uint channel, uint3 threadIdx, float value)
{
	sdata[channel][threadIdx.z][threadIdx.y][threadIdx.x] = value;
}

float reduce(uint channel, uint3 threadIdx)
{
	return
		max(
			max(
				max(
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 1]
				),
				max(
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 1]
				)
			),
			max(
				max(
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 1]
				),
				max(
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 1]
				)
			)
		);
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void velocitySummaryCS(uint3 tidx : SV_DispatchThreadID, uint3 threadIdx : SV_GroupThreadID)
{
	// in this case, each 8x8x8 thread block will summarize 16x16x16 cells
	int3 vidx = DispatchIDToVirtual(tidx);

	// center of 2x2x2 cell cluster
	float3 vidxf = float3(vidx)+0.5f.xxx + 0.5f.xxx;

	float4 vel = sampleVelocity(vidxf);

	// do not use the w channel here
	vel.w = 0.f;

	sdata_write(0, threadIdx, dot(vel, vel));

	GroupMemoryBarrierWithGroupSync();

	// reduce 8x8x8 to 4x4x4 set using shared memory
	if (all(threadIdx < 4))
	{
		float val = reduce(0, threadIdx);
		sdata_write(1, threadIdx, val);
	}

	GroupMemoryBarrierWithGroupSync();

	// reduce 4x4x4 to 2x2x2 set using shared memory
	if (all(threadIdx < 2))
	{
		float val = reduce(1, threadIdx);
		sdata_write(0, threadIdx, val);
	}

	GroupMemoryBarrierWithGroupSync();

	// reduce 2x2x2 to 1x1x1 and write output
	if (all(threadIdx < 1))
	{
		float val = reduce(0, threadIdx);

		// summary is for 16x16x16
		summaryUAV[(vidx >> 4)] = val;
	}
}