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
#define THREAD_DIM_Z 1

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowUint4 gridDim;
	NvFlowUint4 poolGridDim;
};

Texture3D<float> allocMaskSRV : register(t0);

RWBuffer<uint2> shadowBlockList : register(u0);

RWBuffer<uint> atomicUAV : register(u1);

RWTexture3D<uint> shadowBlockTable : register(u2);

uint allocateList()
{
	uint allocIdx;
	InterlockedAdd(atomicUAV[0], 1u, allocIdx);
	return allocIdx;
}

uint allocateTable()
{
	uint allocIdx;
	InterlockedAdd(atomicUAV[1], 1u, allocIdx);
	return allocIdx;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void volumeShadowGenListCS(uint3 tidx : SV_DispatchThreadID)
{
	uint mink = gridDim.z;
	uint maxk = 0u;
	for (uint k = 0; k < gridDim.z; k++)
	{
		int3 maskIdx = int3(tidx.xy, k);

		uint maskVal = allocMaskSRV[maskIdx];

		uint blockTableVal = 0u;

		if (maskVal != 0u)
		{
			mink = min(mink, k);
			maxk = max(maxk, k);

			uint allocIdx = allocateTable();
			
			int3 allocIdx3 = int3(
				allocIdx % poolGridDim.x,
				(allocIdx / poolGridDim.x) % poolGridDim.y,
				allocIdx / (poolGridDim.x * poolGridDim.y)
				);

			if (allocIdx3.z < int(poolGridDim.z))
			{
				blockTableVal = coord_to_tableVal(allocIdx3);
			}
		}

		shadowBlockTable[maskIdx] = blockTableVal;
	}

	if (mink <= maxk)
	{
		uint allocIdx = allocateList();

		uint2 listVal = uint2(
			coord_to_tableVal(int3(tidx.xy, mink)),
			coord_to_tableVal(int3(tidx.xy, maxk))
			);

		shadowBlockList[allocIdx] = listVal;
	}
}