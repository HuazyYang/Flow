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
	NvFlowShaderLinearParams densityParams;

	NvFlowUint4 subBlockDimBits;
	NvFlowFloat4 gridWorldHalfSize;
	NvFlowFloat4 gridWorldLocation;

	NvFlowUint4 blockIdxOffset;

	NvFlowFloat4 velocityScale;
	NvFlowFloat4 densityScale;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! Summary buffer
RWBuffer<float4> summaryUAV : register(u0);

//! Velocity field input
Texture3D<float4> velocitySRV : register(t1);
Texture3D<uint> velocityBlockTable : register(t2);

//! Density field input
Texture3D<float4> densitySRV : register(t3);
Texture3D<uint> densityBlockTable : register(t4);

SAMPLE_LINEAR_3D(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);
SAMPLE_LINEAR_3D(sampleDensity, float4, densitySRV, densityBlockTable, densityParams);

groupshared float4 sdata[2][8][8][8];

void sdata_write(uint channel, uint3 threadIdx, float4 value)
{
	sdata[channel][threadIdx.z][threadIdx.y][threadIdx.x] = value;
}

float4 add(float4 a, float4 b)
{
	return a + b;
}

float4 reduce(uint channel, uint3 threadIdx)
{
	return
		add(
			add(
				add(
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 1]
				),
				add(
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 1]
				)
			),
			add(
				add(
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 1]
				),
				add(
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 0],
					sdata[channel][2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 1]
				)
			)
		);
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridSummaryCS(uint3 tidx : SV_DispatchThreadID, uint3 threadIdx : SV_GroupThreadID)
{
	uint blockID = tidx.x >> velocityParams.blockDimBits.x;
	int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);
	int3 cellIdx = (tidx & (velocityParams.blockDim.xyz - int3(1, 1, 1)));
	int3 vidx = (vBlockIdx << velocityParams.blockDimBits.xyz) | cellIdx;

	// determine which subblock this thread block belongs to
	int3 subBlockIdx = (cellIdx >> int3(3, 3, 3));
	uint bufferIdx = ((blockID + blockIdxOffset.x) << subBlockDimBits.w) |
		(subBlockIdx.z << (subBlockDimBits.x + subBlockDimBits.y)) | (subBlockIdx.y << subBlockDimBits.x) | subBlockIdx.x;

	// center of single cell
	float3 vidxf = float3(vidx)+0.5f.xxx;

	// fetch and reduce velocity
	{
		float4 vel = sampleVelocity(vidxf);

		vel.w = length(vel.xyz);

		sdata_write(0, threadIdx, vel);

		GroupMemoryBarrierWithGroupSync();

		// reduce 8x8x8 to 4x4x4 set using shared memory
		if (all(threadIdx < 4))
		{
			float4 val = reduce(0, threadIdx);
			sdata_write(1, threadIdx, val);
		}

		GroupMemoryBarrierWithGroupSync();

		// reduce 4x4x4 to 2x2x2 set using shared memory
		if (all(threadIdx < 2))
		{
			float4 val = reduce(1, threadIdx);
			sdata_write(0, threadIdx, val);
		}

		GroupMemoryBarrierWithGroupSync();

		// reduce 2x2x2 to 1x1x1 and write output
		if (all(threadIdx < 1))
		{
			float4 val = reduce(0, threadIdx);

			val = val * velocityScale;

			summaryUAV[4 * bufferIdx + 2] = val;
		}
	}

	// Conservative barrier, just to isolate velocity and density pass
	GroupMemoryBarrierWithGroupSync();

	// fetch and reduce density
	{
		float4 density = sampleDensity(vidxf);

		sdata_write(0, threadIdx, density);

		GroupMemoryBarrierWithGroupSync();

		// reduce 8x8x8 to 4x4x4 set using shared memory
		if (all(threadIdx < 4))
		{
			float4 val = reduce(0, threadIdx);
			sdata_write(1, threadIdx, val);
		}

		GroupMemoryBarrierWithGroupSync();

		// reduce 4x4x4 to 2x2x2 set using shared memory
		if (all(threadIdx < 2))
		{
			float4 val = reduce(1, threadIdx);
			sdata_write(0, threadIdx, val);
		}

		GroupMemoryBarrierWithGroupSync();

		// reduce 2x2x2 to 1x1x1 and write output
		if (all(threadIdx < 1))
		{
			float4 val = reduce(0, threadIdx);

			val = val * densityScale;

			summaryUAV[4 * bufferIdx + 3] = val;
		}
	}

	// compute and output the world bounding box for this sample
	if (all(threadIdx < 1))
	{
		int3 sampleGridDim = (velocityParams.gridDim.xyz << subBlockDimBits.xyz);
		int3 sampleBlockIdx = (vBlockIdx << subBlockDimBits.xyz) | subBlockIdx;

		float3 uvw = (float3(sampleBlockIdx)+0.5f.xxx) / float3(sampleGridDim);
		float3 ndc = 2.f.xxx * uvw - 1.f.xxx;

		float3 worldLocation = gridWorldHalfSize.xyz * ndc + gridWorldLocation.xyz;
		float3 worldHalfSize = gridWorldHalfSize.xyz * (1.f / float3(sampleGridDim));

		summaryUAV[4 * bufferIdx + 0] = float4(worldLocation, 1.f);
		summaryUAV[4 * bufferIdx + 1] = float4(worldHalfSize, 0.f);
	}
}