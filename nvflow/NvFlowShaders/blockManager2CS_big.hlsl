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
	#include "../NvFlow/blockManagerShaderParams.h"
};

RWTexture3D<uint> fieldMappingUAV : register(u0);

Texture3D<uint> userMappingSRV : register(t0);
Texture3D<float> velocitySummarySRV : register(t1);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void blockManager2CS_big(uint3 tidx : SV_DispatchThreadID)
{
	int3 mapWriteIdx = tidx;
	int3 mapReadIdx = mapWriteIdx + mapIdxReadOffset.xyz;
	uint accum = 0u;

	// always apply nearest neighbors
	accum |= userMappingSRV[mapReadIdx + int3(-1, 0, 0)];
	accum |= userMappingSRV[mapReadIdx + int3(+1, 0, 0)];
	accum |= userMappingSRV[mapReadIdx + int3(0, -1, 0)];
	accum |= userMappingSRV[mapReadIdx + int3(0, +1, 0)];
	accum |= userMappingSRV[mapReadIdx + int3(0, 0, -1)];
	accum |= userMappingSRV[mapReadIdx + int3(0, 0, +1)];
	
	accum |= userMappingSRV[mapReadIdx];

	const int rx = 3;
	const int ry = 3;
	const int rz = 3;
	for (int k = -rz; k <= rz; k++)
	{
		for (int j = -ry; j <= ry; j++)
		{
			for (int i = -rx; i <= rx; i++)
			{
				int3 tempMapIdx = mapReadIdx + int3(i, j, k);
				uint tempAccum = userMappingSRV[tempMapIdx];
				if (tempAccum & 0x07)
				{
					float vel2 = velocitySummarySRV[tempMapIdx >> velFactorBits.xyz];
					float predictRadius = velocityScale * sqrt(vel2);
					float dist2 = float(i * i) + float(j * j) + float(k * k);
					if (dist2 < (predictRadius * predictRadius))
					{
						accum |= tempAccum;
					}
				}
			}
		}
	}

	fieldMappingUAV[mapWriteIdx] = accum;
}