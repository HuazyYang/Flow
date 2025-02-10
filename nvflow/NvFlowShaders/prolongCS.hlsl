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
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float2> pressureUAV : register(u0);
Texture3D<uint> pressureBlockTable : register(t1);

//! Input pressure
Texture3D<float2> pressureSRV : register(t2);

//! Pressure from higher level
Texture3D<float2> pressureHSRV : register(t3);

SAMPLE_POINT_3D(samplePressure, float2, pressureSRV, pressureBlockTable, pressureParams);
SAMPLE_POINT_3D(samplePressureH, float2, pressureHSRV, pressureBlockTable, pressureParams);

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
	return samplePressure(level_to_root_out(vidx));
}

float2 samplePressureHLevel(int3 vidx)
{
	return samplePressureH(level_to_root_in(vidx));
}

void outputPressureLevel(int3 vidx, float2 value)
{
	outputPressure(level_to_root_out(vidx), value);
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void prolongCS(uint3 tidx : SV_DispatchThreadID)
{
	// vidx at output level
	int3 vidx = DispatchIDToVirtual(tidx);

	// c000 is at input level
	int3 c = vidx - int3(1, 1, 1);
	int3 c000 = c >> 1;

	float p000 = samplePressureHLevel(c000 + int3(0, 0, 0)).x;
	float p100 = samplePressureHLevel(c000 + int3(1, 0, 0)).x;
	float p010 = samplePressureHLevel(c000 + int3(0, 1, 0)).x;
	float p110 = samplePressureHLevel(c000 + int3(1, 1, 0)).x;
	float p001 = samplePressureHLevel(c000 + int3(0, 0, 1)).x;
	float p101 = samplePressureHLevel(c000 + int3(1, 0, 1)).x;
	float p011 = samplePressureHLevel(c000 + int3(0, 1, 1)).x;
	float p111 = samplePressureHLevel(c000 + int3(1, 1, 1)).x;

	float fx = bool(c.x & 1) ? 0.75f : 0.25f;
	float fy = bool(c.y & 1) ? 0.75f : 0.25f;
	float fz = bool(c.z & 1) ? 0.75f : 0.25f;
	float ofx = 1.f - fx;
	float ofy = 1.f - fy;
	float ofz = 1.f - fz;

	float pcorr = ofz*(ofy*(ofx*p000 + fx*p100) + fy*(ofx*p010 + fx*p110)) +
		fz*(ofy*(ofx*p001 + fx*p101) + fy*(ofx*p011 + fx*p111));

	float2 pd = samplePressureLevel(vidx);

	pd.x += pcorr;

	// hack
	//float2 pd = float2(0.f, pcorr);

	outputPressureLevel(vidx, pd);
}