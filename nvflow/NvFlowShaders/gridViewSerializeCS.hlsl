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

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

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

//! The value to serialize
Texture3D<float4> valueSRV : register(t1);
Texture3D<uint> valueBlockTable : register(t2);

RWTexture2D<float2> outputUAV : register(u0);

SAMPLE_LINEAR_3D(sampleValue, float4, valueSRV, valueBlockTable, valueParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void gridViewSerializeCS(uint3 tidx : SV_DispatchThreadID)
{
	// this should be similar to framework.hlsli's version
	uint blockID = tidx.x >> valueParams.blockDimBits.x;
	int3 vBlockIdx = tableVal_to_coord(blockListSRV[blockID]);
	int3 cellIdx = (tidx & (valueParams.blockDim.xyz - int3(1, 1, 1)));
	int3 vidx = (vBlockIdx << valueParams.blockDimBits.xyz) | cellIdx;

	// write out raw data
	float4 data = sampleValue(float3(vidx) + 0.5f.xxx);
	float2 lower = float2(data.x, data.w);

	// 1D cell idx
	uint cellIdx1D =
		(cellIdx.x) |
		(cellIdx.y << valueParams.blockDimBits.x) |
		(cellIdx.z << (valueParams.blockDimBits.x + valueParams.blockDimBits.y));

	// cells per block
	uint blockDimBits1D =
		valueParams.blockDimBits.x +
		valueParams.blockDimBits.y +
		valueParams.blockDimBits.z;

	uint idx1D = ((blockID + blockStart) << blockDimBits1D) + cellIdx1D;

	int2 coord = int2(
		idx1D & (dataWidth - 1),
		idx1D >> dataWidthBits
		);

	outputUAV[coord] = lower;
}