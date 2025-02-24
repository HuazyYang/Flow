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

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams params;
};

Buffer<uint> blockListSRV : register(t0);

Texture3D<float4> valueSRV : register(t1);
Texture3D<uint> blockTableSRV : register(t2);

RWTexture3D<float4> valueUAV : register(u0);

VIRTUAL_TO_REAL(VirtualToReal, blockTableSRV, params)

DISPATCH_ID_TO_VIRTUAL_AND_REAL(blockListSRV, blockTableSRV, params)

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void updateLinearCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 srcVidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, srcVidx, ridx);

	if (srcVidx.x >= 0)
	{
		int3 srcRidx = VirtualToReal(srcVidx);

		float4 value = valueSRV[srcRidx];

		valueUAV[ridx] = value;
	}
}