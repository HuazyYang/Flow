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
	NvFlowShaderLinearParams linearParams;
	NvFlowUint dispatchNumBlocks;
	NvFlowUint padd1;
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

RWTexture3D<float2> fieldUAV : register(u0);

Texture3D<float2> fieldSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);
Buffer<uint> blockListSRV : register(t2);

VIRTUAL_TO_REAL(VirtualToReal, blockTableSRV, linearParams);

DISPATCH_ID_TO_VIRTUAL(blockListSRV, linearParams);

void execute(uint3 tidx)
{
	int3 vidx = DispatchIDToVirtual(tidx);
	int3 ridx = VirtualToReal(vidx);

	float2 sum = 0.f.xx;

	float2 oldVal = fieldSRV[ridx];

	sum += fieldSRV[VirtualToReal(vidx + int3(-1, 0, 0))];
	sum += fieldSRV[VirtualToReal(vidx + int3(+1, 0, 0))];
	sum += fieldSRV[VirtualToReal(vidx + int3(0, -1, 0))];
	sum += fieldSRV[VirtualToReal(vidx + int3(0, +1, 0))];
	sum += fieldSRV[VirtualToReal(vidx + int3(0, 0, -1))];
	sum += fieldSRV[VirtualToReal(vidx + int3(0, 0, +1))];

	sum += 2.f * oldVal;

	sum *= (1.f / 8.f);

	sum.y = 0.f;

	//if (oldVal.y > 0.f)
	//{
	//	sum = oldVal;
	//}

	fieldUAV[ridx] = sum;
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceSmoothCS(uint3 tidx : SV_DispatchThreadID)
{
	uint blockID = tidx.x >> linearParams.blockDimBits.x;
	for (; blockID < blockListLen; blockID += dispatchNumBlocks)
	{
		execute(tidx);
		tidx += (dispatchNumBlocks << linearParams.blockDimBits.x);
	}
}