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

Texture3D<float> blockDistanceFieldSRV : register(t2);

Texture3D<uint2> rangeListSRV : register(t3);
Buffer<uint> cellListSRV : register(t4);

Texture3D<float> signFieldSRV : register(t5);

Buffer<uint> blockListSRV : register(t6);

RWTexture3D<float> distanceFieldUAV : register(u0);

uint3 decode(uint val)
{
	uint valInv = ~val;
	return uint3(
		(valInv >> 0) & 0x3FF,
		(valInv >> 10) & 0x3FF,
		(valInv >> 20) & 0x3FF
		);
}

uint distSq(uint3 a, uint3 b)
{
	uint3 range = int3(a)-int3(b);
	uint dist2 = dot(range, range);
	return dist2;
}

float cellDist2(uint3 a, uint3 b)
{
	return float(distSq(a, b));
}

groupshared uint4 s_blockList[BLOCK_DIM*BLOCK_DIM*BLOCK_DIM];
groupshared uint4 s_cellList[2][BLOCK_DIM*BLOCK_DIM*BLOCK_DIM];
groupshared uint s_batchCoverageAtomic[2];

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)]
void unsignedDistanceCS(uint3 cellIdx : SV_DispatchThreadID, uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	// start with a very large search distance, units in cells
	uint cellMaxDist = ~0u;
	float cellMinDist2 = 1e9f;

	const float ep = 0.01f;
	float nearestBlockDist = blockDistanceFieldSRV[blockIdx.xyz];
	float blockDistMax = nearestBlockDist + 1.5f;
	float blockDistMin2 = nearestBlockDist * nearestBlockDist - ep;
	float blockDistMax2 = blockDistMax * blockDistMax;

	uint3 threadIdx1D = uint3(threadIdx.z * BLOCK_DIM * BLOCK_DIM + threadIdx.y * BLOCK_DIM + threadIdx.x, 1, 1);

	//uint cellCount = 0u;

	// initialize atomicss
	if (all(threadIdx.xyz == uint3(0, 0, 0)))
	{
		s_batchCoverageAtomic[0] = 0u;
		s_batchCoverageAtomic[1] = 0u;
	}

	// RAW hazard: s_batchCoverageAtomic
	GroupMemoryBarrierWithGroupSync();

	// search in BLOCK_DIM x BLOCK_DIM block batches
	uint batchFlip = 0u;
	uint blockFlip = 0u;

	//for (uint blockListIdx = 0u; blockListIdx < blockCount; blockListIdx += BLOCK_DIM_1D)
	uint bcount = (blockCount + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D;
	for (uint bidx = 0u; bidx < bcount; bidx++)
	{
		uint blockListIdx = bidx *  BLOCK_DIM_1D + threadIdx1D.x;
		// check batch blocks for the presence of a boundary condition, and produce a compacted list
		if (blockListIdx < blockCount)
		{
			uint3 testBlockIdx = decode(blockListSRV[blockListIdx]);

			// check if block is within blockMaxDist
			uint testBlockDist2 = distSq(blockIdx.xyz, testBlockIdx);
			if (float(testBlockDist2) >= blockDistMin2 && float(testBlockDist2) <= blockDistMax2)
			{
				float4 normal = blockNormalFieldSRV[testBlockIdx];
				if (normal.w != 0.f)
				{
					uint allocIdx;
					InterlockedAdd(s_batchCoverageAtomic[batchFlip], 1u, allocIdx);
					s_blockList[allocIdx] = uint4(testBlockIdx, 0u);
				}
			}
		}

		// reset atomic for next batch
		if (all(threadIdx.xyz == uint3(0, 0, 0)))
		{
			s_batchCoverageAtomic[batchFlip ^ 1] = 0u;
		}

		// wait for complete list
		GroupMemoryBarrierWithGroupSync();

		// process interaction between this block and each test block
		uint numTestBlocks = s_batchCoverageAtomic[batchFlip];
		for (uint blockListIndex = 0u; blockListIndex < numTestBlocks; blockListIndex++)
		{
			uint3 testBlockIdx = s_blockList[blockListIndex].xyz;

			// look up offset and range in compacted list
			uint2 rangeListVal = rangeListSRV[testBlockIdx];
			uint cellFetchOffset = rangeListVal.x;
			uint numCells = rangeListVal.y;

			// do a parallel cell fetch
			if (threadIdx1D.x < numCells)
			{
				uint testCellIdxVal = ~cellListSRV[threadIdx1D.x + cellFetchOffset];
				uint4 testCellIdx = uint4(
					(testCellIdxVal >> 0) & 0x3FF,
					(testCellIdxVal >> 10) & 0x3FF,
					(testCellIdxVal >> 20) & 0x3FF,
					0u
					);

				testCellIdx.xyz = testBlockIdx * BLOCK_DIM + testCellIdx.xyz;

				s_cellList[blockFlip][threadIdx1D.x] = testCellIdx;
			}

			GroupMemoryBarrierWithGroupSync();

			//cellCount++;

			// test for closest boundary condition
			for (uint cellListIndex = 0u; cellListIndex < numCells; cellListIndex++)
			{
				uint3 testCellIdx = s_cellList[blockFlip][cellListIndex].xyz;

				// test fine estimate
				float testDist2 = cellDist2(cellIdx.xyz, testCellIdx.xyz);
				if (testDist2 < cellMinDist2)
				{
					cellMinDist2 = testDist2;
				}
			}

			// WAR hazard s_cellList
			blockFlip ^= 1;
		}

		// flip
		batchFlip ^= 1;

		// TODO: figure out why the compiler demands this sync
		GroupMemoryBarrierWithGroupSync();
	}

	// write out minimum distance
	float dist = sqrt(cellMinDist2);
	dist *= sign(signFieldSRV[cellIdx.xyz]);

	dist *= scale.x;

	//if (cellIdx.y < 8u) dist = 0.f;

	//dist = (1.f / 64.f) * float(cellCount);

	distanceFieldUAV[cellIdx.xyz] = dist;
}