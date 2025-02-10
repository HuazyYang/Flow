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

#include "../NvFlowShaders/framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowUint4 gridDim;
	NvFlowUint4 poolGridDim;
};

RWTexture3D<uint> blockTableUAV : register(u0);
RWBuffer<uint> blockListUAV : register(u1);
RWBuffer<uint> atomicUAV : register(u2);

uint allocate()
{
	uint allocIdx;
	InterlockedAdd(atomicUAV[0], 1u, allocIdx);
	return allocIdx;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceBlockTableAllocCS(uint3 blockTableIdx : SV_DispatchThreadID)
{
	uint blockTableVal = blockTableUAV[blockTableIdx];
	if (blockTableVal != 0u)
	{
		uint allocIdx = allocate();

		int3 allocIdx3 = int3(
			allocIdx % poolGridDim.x,
			(allocIdx / poolGridDim.x) % poolGridDim.y,
			allocIdx / (poolGridDim.x * poolGridDim.y)
			);

		if (allocIdx3.z < int(poolGridDim.z))
		{
			blockTableUAV[blockTableIdx] = coord_to_tableVal(allocIdx3);
			blockListUAV[allocIdx] = coord_to_tableVal(blockTableIdx);
		}
	}	
}