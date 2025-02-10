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

RWBuffer<uint> freeListUAV : register(u0);
RWBuffer<uint> atomicUAV : register(u1);

Texture3D<uint2> frameTableSRV : register(t0);

uint allocate()
{
	uint allocIdx;
	InterlockedAdd(atomicUAV[0], 1u, allocIdx);
	return allocIdx;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseFreeListCS(uint3 frameTableIdx : SV_DispatchThreadID)
{
	uint2 frameTableVal = frameTableSRV[frameTableIdx];

	// condition for free
	// must check bounds because out of bounds frame table fetches return 0
	if (frameTableVal.x == 0u && all(frameTableIdx < frameTableDim.xyz))
	{
		uint idx = allocate();

		freeListUAV[idx] = coord_to_tableVal(frameTableIdx);
	}
}