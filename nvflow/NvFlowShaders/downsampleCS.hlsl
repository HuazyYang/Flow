/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define BLOCK_DIM 4

Texture3D<float4> normalFieldSRV : register(t0);

RWTexture3D<float4> normalField2UAV : register(u0);
RWTexture3D<float4> normalField4UAV : register(u1);
RWTexture3D<float4> normalField8UAV : register(u2);

RWTexture3D<float> signField32UAV : register(u3);

groupshared float4 sdata0[BLOCK_DIM][BLOCK_DIM][BLOCK_DIM];
groupshared float4 sdata1[BLOCK_DIM][BLOCK_DIM][BLOCK_DIM];

void normalCleanup(inout float4 normal)
{
	// if there is one normal, normal.w is already > 0.25f
	if (normal.w > 0.1f)
	{
		normal.w = 1.f;
		normal.xyz = normalize(normal.xyz);
	}
}

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)]
void downsampleCS(uint3 tidx : SV_DispatchThreadID, uint3 blockIdx : SV_GroupID, uint3 threadIdx : SV_GroupThreadID)
{
	float4 normal;

	normal = normalFieldSRV[2u * tidx.xyz + uint3(0, 0, 0)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(1, 0, 0)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(0, 1, 0)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(1, 1, 0)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(0, 0, 1)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(1, 0, 1)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(0, 1, 1)];
	normal += normalFieldSRV[2u * tidx.xyz + uint3(1, 1, 1)];

	normal *= 0.125f;
	normalCleanup(normal);

	normalField2UAV[tidx.xyz] = normal;
	sdata0[threadIdx.z][threadIdx.y][threadIdx.x] = normal;

	// RAW hazard: sdata0
	GroupMemoryBarrierWithGroupSync();

	if (all(threadIdx.xyz < uint3(BLOCK_DIM / 2, BLOCK_DIM / 2, BLOCK_DIM / 2)))
	{
		normal = sdata0[2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 0];
		normal += sdata0[2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 1];
		normal += sdata0[2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 0];
		normal += sdata0[2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 1];
		normal += sdata0[2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 0];
		normal += sdata0[2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 1];
		normal += sdata0[2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 0];
		normal += sdata0[2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 1];

		normal *= 0.125f;
		normalCleanup(normal);

		uint3 cellIdx = blockIdx.xyz * uint3(BLOCK_DIM / 2, BLOCK_DIM / 2, BLOCK_DIM / 2) + threadIdx.xyz;
		normalField4UAV[cellIdx] = normal;
		sdata1[threadIdx.z][threadIdx.y][threadIdx.x] = normal;
	}

	// RAW hazard: sdata1
	GroupMemoryBarrierWithGroupSync();

	if (all(threadIdx.xyz < uint3(BLOCK_DIM / 4, BLOCK_DIM / 4, BLOCK_DIM / 4)))
	{
		normal = sdata1[2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 0];
		normal += sdata1[2 * threadIdx.z + 0][2 * threadIdx.y + 0][2 * threadIdx.x + 1];
		normal += sdata1[2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 0];
		normal += sdata1[2 * threadIdx.z + 0][2 * threadIdx.y + 1][2 * threadIdx.x + 1];
		normal += sdata1[2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 0];
		normal += sdata1[2 * threadIdx.z + 1][2 * threadIdx.y + 0][2 * threadIdx.x + 1];
		normal += sdata1[2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 0];
		normal += sdata1[2 * threadIdx.z + 1][2 * threadIdx.y + 1][2 * threadIdx.x + 1];

		normal *= 0.125f;
		normalCleanup(normal);

		normalField8UAV[blockIdx.xyz] = normal;

		// initialize sign field
		signField32UAV[blockIdx.xyz] = 0.f;
	}
}