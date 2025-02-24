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

cbuffer params : register(b0)
{
	NvFlowUint4 frameTableDim;
	NvFlowUint4 layerIdx;
};

RWTexture3D<uint> blockTableUAV : register(u0);
RWTexture3D<uint2> frameTableUAV : register(u1);
RWBuffer<uint> atomicUAV : register(u2);
RWTexture3D<uint> blockTable1DUAV : register(u3);

Texture3D<uint> maskSRV : register(t0);
Buffer<uint> freeListSRV : register(t1);

uint allocate()
{
	uint allocIdx;
	InterlockedAdd(atomicUAV[1], 1u, allocIdx);
	return allocIdx;
}

uint freeListSize()
{
	return atomicUAV[0];
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseAllocateCS(uint3 blockTableIdx : SV_DispatchThreadID)
{
	uint blockTableVal = blockTableUAV[blockTableIdx];
	uint maskVal = maskSRV[blockTableIdx];

	// condition for allocation
	// mask will never be non-zero with an out of bounds fetch
	if (maskVal != 0 && blockTableVal == 0)
	{
		uint freeListIdx = allocate();

		if (freeListIdx < freeListSize())
		{
			uint frameID = freeListSRV[freeListIdx];
			int3 frameIdx = tableVal_to_coord(frameID);

			blockTableVal = frameID;
			blockTableUAV[blockTableIdx] = blockTableVal;
			frameTableUAV[frameIdx] = uint2(coord_to_tableVal(blockTableIdx), layerIdx.x);
		}
	}

	// update 1D blocktable
	{
		uint3 rBlockIdx = uint3(tableVal_to_coord(blockTableVal));
		uint tileID = 0u;
		if (all(rBlockIdx < frameTableDim.xyz))
		{
			tileID = (rBlockIdx.z * frameTableDim.y + rBlockIdx.y) * frameTableDim.x + rBlockIdx.x;
			tileID = ~tileID;
		}
		blockTable1DUAV[blockTableIdx] = tileID;
	}
}