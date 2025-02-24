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

#define THREAD_DIM_X 128
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

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
Buffer<uint> blockListSRV : register(t1);

//! The value to serialize
RWTexture3D<float4> valueUAV : register(u0);
Texture3D<uint> valueBlockTable : register(t2);
Texture3D<uint> vBlockIdxToBlockID : register(t3);

Texture2D<float2> inputSRV : register(t0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridViewDeserializeCS(uint3 tidx : SV_DispatchThreadID)
{
	uint dstBlockID = tidx.y;
	uint3 dstCellIdx = int3(
		tidx.x % valueParams.linearBlockDim.x,
		(tidx.x / valueParams.linearBlockDim.x) % valueParams.linearBlockDim.y,
		tidx.x / (valueParams.linearBlockDim.x * valueParams.linearBlockDim.y));

	if (dstCellIdx.z < valueParams.linearBlockDim.z)
	{
		int3 dst_vBlockIdx = tableVal_to_coord(blockListSRV[dstBlockID]);

		int3 srcVidx = dst_vBlockIdx * valueParams.blockDim.xyz + dstCellIdx - valueParams.linearBlockOffset.xyz;

		int3 src_vBlockIdx = srcVidx >> valueParams.blockDimBits.xyz;
		uint srcBlockID = ~vBlockIdxToBlockID[src_vBlockIdx];
		int3 srcCellIdx = srcVidx & (valueParams.blockDim.xyz - int3(1, 1, 1));

		// 1D cell idx
		uint cellIdx1D =
			(srcCellIdx.x) |
			(srcCellIdx.y << valueParams.blockDimBits.x) |
			(srcCellIdx.z << (valueParams.blockDimBits.x + valueParams.blockDimBits.y));

		// cells per block
		uint blockDimBits1D =
			valueParams.blockDimBits.x +
			valueParams.blockDimBits.y +
			valueParams.blockDimBits.z;

		uint idx1D = (srcBlockID << blockDimBits1D) + cellIdx1D;

		int2 coord = int2(
			idx1D & (dataWidth - 1),
			idx1D >> dataWidthBits
			);

		// read in raw data
		float2 data = inputSRV[coord];

		// write out raw data
		float4 value;
		value.x = data.x;
		value.y = 0.f;
		value.z = 0.f;
		value.w = data.y;

		// lookup real block index
		int3 rBlockIdx = tableVal_to_coord(valueBlockTable[dst_vBlockIdx]);
		int3 ridx = rBlockIdx * valueParams.linearBlockDim.xyz + dstCellIdx;

		valueUAV[ridx] = value;
	}
}