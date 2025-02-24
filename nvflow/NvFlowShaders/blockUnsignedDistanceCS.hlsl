/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define BLOCK_DIM 8

#define BLOCK_DIM_1D 512

cbuffer params : register(b0)
{
	uint4 blockFieldDim;
	float4 scale;
};

cbuffer params : register(b1)
{
	uint blockCount;
	uint cellCount;
	uint pad1;
	uint pad2;
};

Texture3D<float4> cellNormalFieldSRV : register(t0);
Texture3D<float4> blockNormalFieldSRV : register(t1);

Buffer<uint> blockListSRV : register(t2);

RWTexture3D<float> blockDistanceFieldUAV : register(u0);

groupshared uint s_minBlockDist;

uint3 decode(uint val)
{
	uint valInv = ~val;
	return uint3(
		(valInv >> 0) & 0x3FF,
		(valInv >> 10) & 0x3FF,
		(valInv >> 20) & 0x3FF
		);
}

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)]
void blockUnsignedDistanceCS(uint3 cellIdx : SV_DispatchThreadID, uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	// initialize atomicss
	if (all(threadIdx.xyz == uint3(0, 0, 0)))
	{
		s_minBlockDist = ~0u;
	}

	uint3 threadIdx1D = uint3(threadIdx.z * BLOCK_DIM * BLOCK_DIM + threadIdx.y * BLOCK_DIM + threadIdx.x, 1, 1);

	// RAW hazard: s_minBlockDist
	GroupMemoryBarrierWithGroupSync();

	uint bcount = (blockCount + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
	for (uint bidx = 0u; bidx < bcount; bidx++)
	{
		uint blockListIdx = bidx *  BLOCK_DIM_1D + threadIdx1D.x;
		// check batch blocks for the presence of a boundary condition, and produce a compacted list
		if (blockListIdx < blockCount)
		{
			uint3 testBlockIdx = decode(blockListSRV[blockListIdx]);

			// compute distance to block
			uint3 range = int3(blockIdx.xyz) - int3(testBlockIdx);
			uint dist2 = dot(range, range);

			float4 normal = blockNormalFieldSRV[testBlockIdx];
			if (normal.w != 0.f)
			{
				InterlockedMin(s_minBlockDist, dist2);
			}
		}
	}

	// RAW hazard: s_minBlockDist
	GroupMemoryBarrierWithGroupSync();

	if (all(threadIdx.xyz == uint3(0, 0, 0)))
	{
		// write out minimum distance
		float dist = sqrt(float(s_minBlockDist));

		blockDistanceFieldUAV[blockIdx.xyz] = dist;
	}
}