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

#define EMIT_DEFINES 1

#include "../NvFlow/emitterShaderParams.h"

cbuffer params : register(b0)
{
	NvFlowFloat4x4 gridToWorld;

	NvFlowShaderPointParams valueParams;
	NvFlowFloat4 vDimInv;
	NvFlowUint materialIdx;
	NvFlowUint matPad0;
	NvFlowUint matPad1;
	NvFlowUint matPad2;

	NvFlowFloat4 sdata[EMIT_DATA_SIZE];
	NvFlowFloat4 sshape[EMIT_SHAPE_CACHE_SIZE*EMIT_SHAPE_SIZE];
};

//! Value out field
RWTexture3D<float4> valueUAV : register(u0);
Texture3D<uint> valueBlockTable : register(t0);

//! Value in field
Texture3D<float4> valueSRV : register(t1);

//! SDF field
Texture3D<float> sdf_SRV : register(t2);

VIRTUAL_TO_REAL(VirtualToReal, valueBlockTable, valueParams);

#include "emitterFunctions.hlsli"

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void emitterVelocityCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx = tidx + emit_data_minVidx(sdata).xyz;
	int3 ridx = VirtualToReal(vidx);

	float4 grid_ndc = 2.f * float4(vDimInv.xyz*(vidx + 0.5f.xxx), 1.f) - 1.f;

	float4 value = valueSRV[ridx];

	value = emitVelocityFunction(value, grid_ndc);

	valueUAV[ridx] = value;
}