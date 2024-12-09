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
	NvFlowShaderPointParams velocityParams;

	NvFlowShaderPointParams pressureParams;

	NvFlowFloat4 vdimInv;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float4> velocityUAV : register(u0);
Texture3D<uint> velocityBlockTable : register(t1);

//! velocity read
Texture3D<float4> velocitySRV : register(t2);

//! Input pressure
Texture3D<float2> pressureSRV : register(t3);
Texture3D<uint> pressureBlockTable : register(t4);

//! Fade field
Texture3D<float> fadeFieldSRV : register(t5);

SAMPLE_POINT_3D(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);
SAMPLE_POINT_3D(samplePressure, float2, pressureSRV, pressureBlockTable, pressureParams);

//OUTPUT_3D(outputVelocity, float4, velocityUAV, velocityBlockTable, velocityParams);
//DISPATCH_ID_TO_VIRTUAL(blockListSRV, velocityParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, velocityBlockTable, velocityParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void subtractCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float pxp = samplePressure(vidx + int3(+1, 0, 0)).x;
	float pxn = samplePressure(vidx + int3(-1, 0, 0)).x;
	float pyp = samplePressure(vidx + int3(0, +1, 0)).x;
	float pyn = samplePressure(vidx + int3(0, -1, 0)).x;
	float pzp = samplePressure(vidx + int3(0, 0, +1)).x;
	float pzn = samplePressure(vidx + int3(0, 0, -1)).x;

	float4 uvwa = sampleVelocity(vidx);
	uvwa.x -= 0.5f * (pxp - pxn);
	uvwa.y -= 0.5f * (pyp - pyn);
	uvwa.z -= 0.5f * (pzp - pzn);

	// fade field
	{
		float3 vidxf = float3(vidx)+0.5f.xxx;
		float3 vidxNorm = vdimInv.xyz * vidxf;
		float fadeRate = fadeFieldSRV.SampleLevel(borderSampler, vidxNorm, 0);

		// make the weight zero at or before the block edge
		fadeRate = saturate(2.f * fadeRate - 1.f);

		// override w channel with fadeRate
		uvwa.w = fadeRate;
	}

	velocityUAV[ridx] = uvwa;
	//outputVelocity(vidx, uvwa);
}