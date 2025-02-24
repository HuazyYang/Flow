/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef THREAD_DIM_X

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#define SIMULATE_ONCE 1

#endif

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams outputParams;

	NvFlowShaderLinearParams valueParams;
	NvFlowShaderLinearParams velocityParams;

	NvFlowInt4 vidxOldOffset;
	NvFlowFloat4 deltaTime;
	NvFlowFloat4 damping;
};

//! A list of tiles to perform operation on
Buffer<uint> blockListSRV : register(t0);

//! The target to write out results to
RWTexture3D<float4> valueOutUAV : register(u0);
Texture3D<uint> valueOutBlockTable : register(t1);

//! The value to advect
Texture3D<float4> valueSRV : register(t2);
Texture3D<uint> valueBlockTable : register(t3);

//! The velocity field for advection
Texture3D<float4> velocitySRV : register(t4);
Texture3D<uint> velocityBlockTable : register(t5);

SAMPLE_LINEAR_3D(sampleValue, float4, valueSRV, valueBlockTable, valueParams);
SAMPLE_LINEAR_3D_NORM(sampleVelocity, float4, velocitySRV, velocityBlockTable, velocityParams);

//OUTPUT_3D(outputValue, float4, valueOutUAV, valueOutBlockTable, outputParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, valueOutBlockTable, outputParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void advectCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float3 vidxOldf = float3(vidx + vidxOldOffset.xyz)+0.5f.xxx;

	float4 vel = sampleVelocity(valueParams.vdimInv.xyz * vidxOldf);
	float3 fetchIdxf = vidxOldf - deltaTime.xyz * vel.xyz;

	float4 value = sampleValue(fetchIdxf);

	//// TEST emitter
	//float3 uvw = vidxf - float3(64.f,48.f,64.f);
	//float d2 = dot(uvw, uvw);
	//if (d2 < 64.f)
	//{
	//	value.y += 5.f * (250.f - value.y) * deltaTime.w;
	//	value.w += 5.f * (2.f - value.w) * deltaTime.w;
	//}

	value *= damping;

	valueOutUAV[ridx] = value;
}