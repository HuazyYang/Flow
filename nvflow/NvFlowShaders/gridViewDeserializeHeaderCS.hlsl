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
RWBuffer<uint> blockListUAV : register(u0);

//! The block table of the value to serialize
RWTexture3D<uint> valueBlockTable : register(u1);
RWTexture3D<uint> vBlockIdxToBlockID : register(u2);

Texture2D<float2> inputSRV : register(t0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridViewDeserializeHeaderCS(uint3 tidx : SV_DispatchThreadID)
{
	uint blockID = tidx.x;

	if (blockID < numBlocks)
	{
		int2 coord = int2(
			(blockID + blockStart) & (headerWidth - 1),
			(blockID + blockStart) >> headerWidthBits
			);

		float2 valf = inputSRV[coord];

		uint blockListVal = asuint(valf.x);
		uint blockTableVal = asuint(valf.y);

		int3 vBlockIdx = tableVal_to_coord(blockListVal);

		blockListUAV[blockID] = blockListVal;
		valueBlockTable[vBlockIdx] = blockTableVal;
		vBlockIdxToBlockID[vBlockIdx] = ~(blockID + blockStart);
	}
	else
	{
		blockListUAV[blockID] = 0u;
	}
}