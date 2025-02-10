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

#define THREAD_DIM_X 256
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

cbuffer params : register(b0)
{
	NvFlowShaderPointParams levelParams;
	NvFlowShaderPointParams pressureParams;
	NvFlowFloat4 dx2;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float2> pressureUAV : register(u0);
Texture3D<uint> pressureBlockTable : register(t1);

//! Input pressure
Texture3D<float2> pressureSRV : register(t2);

SAMPLE_POINT_3D(samplePressure, float2, pressureSRV, pressureBlockTable, pressureParams);

OUTPUT_3D(outputPressure, float2, pressureUAV, pressureBlockTable, pressureParams);

DISPATCH_ID_TO_VIRTUAL_1D(blockListSRV, levelParams);

int3 level_to_root(int3 vidx)
{
	// decompose using level scale parameters
	int3 vBlockIdx = vidx >> levelParams.blockDimBits.xyz;
	int3 cellIdx = vidx & (levelParams.blockDim.xyz - int3(1, 1, 1));
	// recompose using root scale parameters
	int3 vidxRoot = (vBlockIdx << pressureParams.blockDimBits.xyz) | cellIdx;
	return vidxRoot;
}

float2 samplePressureLevel(int3 vidx)
{
	return samplePressure(level_to_root(vidx));
}

void outputPressureLevel(int3 vidx, float2 value)
{
	outputPressure(level_to_root(vidx), value);
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void jacobiCS(uint3 tidx : SV_DispatchThreadID)
{
	// vidx at level scale
	int3 vidx = DispatchIDToVirtual(tidx);

	// sample at root scale
	float pxp = samplePressureLevel(vidx + int3(+1, 0, 0)).x;
	float pxn = samplePressureLevel(vidx + int3(-1, 0, 0)).x;
	float pyp = samplePressureLevel(vidx + int3(0, +1, 0)).x;
	float pyn = samplePressureLevel(vidx + int3(0, -1, 0)).x;
	float pzp = samplePressureLevel(vidx + int3(0, 0, +1)).x;
	float pzn = samplePressureLevel(vidx + int3(0, 0, -1)).x;

	float div = samplePressureLevel(vidx).y;

	float p = (pxp + pxn + pyp + pyn + pzp + pzn - dx2.x * div) * (1.f / 6.f);

	float2 pd = float2(p, div);

	outputPressureLevel(vidx, pd);
}