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

#include "../NvFlowShaders/frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderPointParams customEmitParams;
	NvFlowShaderLinearParams surfaceParams;

	NvFlowFloat4x4 gridToField;
	NvFlowFloat4 vdimInv;

	NvFlowFloat4 deltaTime;
	NvFlowFloat4 coupleRate;
	NvFlowFloat4 emitValue;

	float surfaceThreshold;
	float padd1;
	float padd2;
	float padd3;
};

Buffer<uint> blockList : register(t0);
Texture3D<uint> blockTable : register(t1);
Texture3D<float4> dataSRV : register(t2);
RWTexture3D<float4> dataUAV : register(u0);

Texture3D<float2> fieldSRV : register(t3);
Texture3D<uint> fieldBlockTableSRV : register(t4);

DISPATCH_ID_TO_VIRTUAL(blockList, customEmitParams);

VIRTUAL_TO_REAL(VirtualToReal, blockTable, customEmitParams);

SAMPLE_LINEAR_3D_NORM(sampleField, float2, fieldSRV, fieldBlockTableSRV, surfaceParams);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceEmitEmitCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx = DispatchIDToVirtual(tidx);
	float3 vidxNDC = 2.f * vdimInv.xyz * (float3(vidx)+0.5f.xxx) - 1.f.xxx;

	int3 ridx = VirtualToReal(vidx);

	float4 value = dataSRV[ridx];

	float4 fieldUVW = 0.5f * mul(float4(vidxNDC, 1.f), gridToField) + 0.5f.xxxx;

	float2 fieldVal = sampleField(fieldUVW.xyz);

	if (fieldVal.x > surfaceThreshold)
	{
		float4 valueRate = saturate(deltaTime * coupleRate);

		value += valueRate * (emitValue - value);
	}

	dataUAV[ridx] = value;
}