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

Texture3D<uint> maskSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseDeallocateCS(uint3 blockTableIdx : SV_DispatchThreadID)
{
	uint blockTableVal = blockTableSRV[blockTableIdx];
	uint maskVal = maskSRV[blockTableIdx];

	// condition for deallocation
	// block table will never be non-zero with an out of bounds fetch
	if (maskVal == 0 && blockTableVal != 0)
	{
		int3 rBlockIdx = tableVal_to_coord(blockTableVal);

		frameTableUAV[rBlockIdx] = uint2(0u, 0u);
		blockTableVal = 0u;
	}

	blockTableUAV[blockTableIdx] = blockTableVal;
}