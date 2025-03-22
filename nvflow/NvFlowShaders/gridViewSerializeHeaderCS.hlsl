/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define THREAD_DIM_X 64
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams valueParams;

	NvFlowUint headerWidth;
	NvFlowUint headerHeight;
	NvFlowUint dataWidth;
	NvFlowUint dataHeight;

	NvFlowUint headerWidthBits;
	NvFlowUint dataWidthBits;
	NvFlowUint numBlocks;
	NvFlowUint blockStart;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The block table of the value to serialize
Texture3D<uint> valueBlockTable : register(t1);

RWTexture2D<float2> outputUAV : register(u0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridViewSerializeHeaderCS( uint3 tidx : SV_DispatchThreadID )
{
	uint blockID = tidx.x;
	if (blockID < numBlocks)
	{
		uint blockListVal = blockListSRV[blockID];

		int3 vBlockIdx = tableVal_to_coord(blockListVal);

		uint2 valui = uint2(
			(blockID < numBlocks) ? blockListVal : 0u,
			(blockID < numBlocks) ? valueBlockTable[vBlockIdx] : 0u
			);

		float2 valf = asfloat(valui);

		int2 coord = int2(
			(blockID + blockStart) & (headerWidth - 1),
			(blockID + blockStart) >> headerWidthBits
			);

		outputUAV[coord] = valf;
	}
}