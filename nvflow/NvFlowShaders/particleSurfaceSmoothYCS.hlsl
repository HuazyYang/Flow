/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define THREAD_DIM_X 4
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 4

#include "../NvFlowShaders/frameworkHybrid.hlsli"

#define KERNEL_HALF_DIM 8
#define KERNEL_DIM (2 * KERNEL_HALF_DIM + 1)

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams linearParams;
	NvFlowUint dispatchNumBlocks;
	NvFlowUint padd1;
	NvFlowUint padd2;
	NvFlowUint padd3;
	NvFlowFloat4 kernel[(KERNEL_DIM + 3) / 4];
};

cbuffer params : register(b1)
{
	uint blockListLen;
	uint padb0;
	uint padb1;
	uint padb2;
};

groupshared float2 sdata[THREAD_DIM_Z][3 * THREAD_DIM_Y][THREAD_DIM_X];

RWTexture3D<float2> fieldUAV : register(u0);

Texture3D<float2> fieldSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);
Buffer<uint> blockListSRV : register(t2);

void sdata_write(int3 idx, float2 value)
{
	sdata[idx.z][idx.y][idx.x] = value;
}

float2 sdata_read(int3 idx)
{
	return sdata[idx.z][idx.y][idx.x];
}

VIRTUAL_TO_REAL(VirtualToReal, blockTableSRV, linearParams);

DISPATCH_ID_TO_VIRTUAL(blockListSRV, linearParams);

void execute(uint3 tidx, uint3 threadIdx)
{
	int3 vidx = DispatchIDToVirtual(tidx);
	int3 ridx = VirtualToReal(vidx);

	float2 sum = 0.f.xx;

	// prefetch to shared memory
	sdata_write(threadIdx + int3(0, 0 * THREAD_DIM_Y, 0), fieldSRV[VirtualToReal(vidx + int3(0, -THREAD_DIM_Y, 0))]);
	sdata_write(threadIdx + int3(0, 1 * THREAD_DIM_Y, 0), fieldSRV[ridx]);
	sdata_write(threadIdx + int3(0, 2 * THREAD_DIM_Y, 0), fieldSRV[VirtualToReal(vidx + int3(0, +THREAD_DIM_Y, 0))]);

	GroupMemoryBarrierWithGroupSync();

	for (int offset = -KERNEL_HALF_DIM; offset <= KERNEL_HALF_DIM; offset++)
	{
		uint kernelIdx = offset + KERNEL_HALF_DIM;
		float4 weights = kernel[kernelIdx / 4u];
		float weight = 0.f;
		if ((kernelIdx & 3) == 0) weight = weights.x;
		if ((kernelIdx & 3) == 1) weight = weights.y;
		if ((kernelIdx & 3) == 2) weight = weights.z;
		if ((kernelIdx & 3) == 3) weight = weights.w;

		sum += weight * sdata_read(threadIdx + int3(0, offset + THREAD_DIM_Y, 0));
	}

	fieldUAV[ridx] = sum;

	GroupMemoryBarrierWithGroupSync();
}

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void particleSurfaceSmoothYCS(uint3 tidx : SV_DispatchThreadID, uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	uint blockID = (blockIdx.x * THREAD_DIM_X) >> linearParams.blockDimBits.x;
	for (; blockID < blockListLen; blockID += dispatchNumBlocks)
	{
		execute(tidx, threadIdx);
		tidx += (dispatchNumBlocks << linearParams.blockDimBits.x);
	}
}