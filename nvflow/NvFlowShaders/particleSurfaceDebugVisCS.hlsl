/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define THREAD_DIM_X 128
#define THREAD_DIM_Y 1
#define THREAD_DIM_Z 1

#define SIMULATE_REDUNDANT 1
#define ENABLE_SST 1
#define ENABLE_VTR 0

#include "../NvFlowShaders/frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams linearParams;
	NvFlowUint dispatchNumBlocks;
	float surfaceThreshold;
	NvFlowUint padd2;
	NvFlowUint padd3;
}

cbuffer params : register(b1)
{
	uint blockListLen;
	uint padb0;
	uint padb1;
	uint padb2;
};

RWTexture3D<float4> debugVisUAV : register(u0);

Texture3D<float2> fieldSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);
Buffer<uint> blockListSRV : register(t2);

void accumSample(inout float4 color, float w, int3 tidx)
{
	float2 val = fieldSRV[tidx];

	if (val.x > surfaceThreshold)
	{
		color += w * float4(1.f, 1.f, 1.f, 1.f);
	}
}

VIRTUAL_TO_REAL_LINEAR(VirtualToReal, blockTableSRV, linearParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, blockTableSRV, linearParams)

void execute(uint3 tidx)
{
	int3 srcVidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, srcVidx, ridx);

	if (srcVidx.x >= 0)
	{
		float3 vidxf = float3(srcVidx)+0.5f.xxx;

		float4 color = float4(0.f, 0.f, 0.f, 0.f);

		accumSample(color, 2.f, VirtualToReal(vidxf + float3(0.f, 0.f, 0.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(-1.f, 0.f, 0.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(+1.f, 0.f, 0.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(0.f, -1.f, 0.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(0.f, +1.f, 0.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(0.f, 0.f, -1.f)));
		accumSample(color, 1.f, VirtualToReal(vidxf + float3(0.f, 0.f, +1.f)));

		color.rgba *= (1.f / 8.f);

		//color.rgba += 0.1f;

		debugVisUAV[ridx] = color;
	}
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceDebugVisCS(uint3 tidx : SV_DispatchThreadID)
{
	uint blockID = tidx.y;
	for (; blockID < blockListLen; blockID += dispatchNumBlocks)
	{
		execute(tidx);
		tidx.y += dispatchNumBlocks;
	}
}