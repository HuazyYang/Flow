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
	NvFlowShaderPointParams outLevelParams;
	NvFlowShaderPointParams inLevelParams;
	NvFlowShaderPointParams pressureParams;
	NvFlowFloat4 dx2Inv;
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

DISPATCH_ID_TO_VIRTUAL_1D(blockListSRV, outLevelParams);

int3 level_to_root_in(int3 vidx)
{
	// decompose using level scale parameters
	int3 vBlockIdx = vidx >> inLevelParams.blockDimBits.xyz;
	int3 cellIdx = vidx & (inLevelParams.blockDim.xyz - int3(1, 1, 1));
	// recompose using root scale parameters
	int3 vidxRoot = (vBlockIdx << pressureParams.blockDimBits.xyz) | cellIdx;
	return vidxRoot;
}

int3 level_to_root_out(int3 vidx)
{
	// decompose using level scale parameters
	int3 vBlockIdx = vidx >> outLevelParams.blockDimBits.xyz;
	int3 cellIdx = vidx & (outLevelParams.blockDim.xyz - int3(1, 1, 1));
	// recompose using root scale parameters
	int3 vidxRoot = (vBlockIdx << pressureParams.blockDimBits.xyz) | cellIdx;
	return vidxRoot;
}

float2 samplePressureLevel(int3 vidx)
{
	return samplePressure(level_to_root_in(vidx));
}

void outputPressureLevel(int3 vidx, float2 value)
{
	outputPressure(level_to_root_out(vidx), value);
}

float residualDevice(int3 vidx)
{
	float pxp = samplePressureLevel(vidx + int3(+1, 0, 0)).x;
	float pxn = samplePressureLevel(vidx + int3(-1, 0, 0)).x;
	float pyp = samplePressureLevel(vidx + int3(0, +1, 0)).x;
	float pyn = samplePressureLevel(vidx + int3(0, -1, 0)).x;
	float pzp = samplePressureLevel(vidx + int3(0, 0, +1)).x;
	float pzn = samplePressureLevel(vidx + int3(0, 0, -1)).x;

	float2 pd = samplePressureLevel(vidx);

	float r = -dx2Inv.x * ((6.f * pd.x) - (pxp + pxn + pyp + pyn + pzp + pzn)) + pd.y;
	return r;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void restrictCS(uint3 tidx : SV_DispatchThreadID)
{
	// out level is 2x smaller than in level
	int3 vidx = DispatchIDToVirtual(tidx) << 1;

	float r = 0.f;
	r += residualDevice(vidx + int3(0, 0, 0));
	r += residualDevice(vidx + int3(1, 0, 0));
	r += residualDevice(vidx + int3(0, 1, 0));
	r += residualDevice(vidx + int3(1, 1, 0));
	r += residualDevice(vidx + int3(0, 0, 1));
	r += residualDevice(vidx + int3(1, 0, 1));
	r += residualDevice(vidx + int3(0, 1, 1));
	r += residualDevice(vidx + int3(1, 1, 1));
	r *= 0.125f;

	float2 pd = float2(0.f, r);

	outputPressureLevel(vidx >> 1, pd);
}