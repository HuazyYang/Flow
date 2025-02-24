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

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowUint4 factor;
	NvFlowUint4 factorBits;
};

RWTexture3D<uint> maskUAV : register(u0);

Texture3D<uint> srcMaskSRV : register(t0);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseScaleCS(uint3 tidx : SV_DispatchThreadID)
{
	uint3 srcIdx = tidx << factorBits.xyz;

	uint accum = 0u;
	for (uint k = 0; k < factor.z; k++)
	{
		for (uint j = 0; j < factor.y; j++)
		{
			for (uint i = 0; i < factor.x; i++)
			{
				uint3 fetchIdx = srcIdx + uint3(i, j, k);
				accum |= srcMaskSRV[fetchIdx];
			}
		}
	}

	maskUAV[tidx] = accum;
}