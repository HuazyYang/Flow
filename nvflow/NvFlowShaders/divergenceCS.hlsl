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

#define SIMULATE_ONCE 1

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderPointParams pressureParams;

	NvFlowShaderPointParams velocityParams;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float2> pressureUAV : register(u0);
Texture3D<uint> pressureBlockTable : register(t1);

//! Input velocity
Texture3D<float4> velocitySRV : register(t2);
Texture3D<uint> velocityBlockTable : register(t3);

SAMPLE_POINT_3D(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);

//OUTPUT_3D(outputPressure, float2, pressureUAV, pressureBlockTable, pressureParams);
//DISPATCH_ID_TO_VIRTUAL(blockListSRV, pressureParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, pressureBlockTable, pressureParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void divergenceCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float vxp = sampleVelocity((vidx + int3(+1, 0, 0))).x;
	float vxn = sampleVelocity((vidx + int3(-1, 0, 0))).x;
	float vyp = sampleVelocity((vidx + int3(0, +1, 0))).y;
	float vyn = sampleVelocity((vidx + int3(0, -1, 0))).y;
	float vzp = sampleVelocity((vidx + int3(0, 0, +1))).z;
	float vzn = sampleVelocity((vidx + int3(0, 0, -1))).z;

	float d = 0.5f * (vxp - vxn + vyp - vyn + vzp - vzn);

	// TODO: make this optional
	float dext = sampleVelocity(vidx).w;
	d -= dext;
	//d -= divergenceExternalSRV[vidx].x;

	pressureUAV[ridx] = float2(0.f, d);
	//outputPressure(vidx, float2(0.f, d));
}