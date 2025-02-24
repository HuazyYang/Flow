/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define BLOCK_DIM 8

cbuffer params : register(b0)
{
	uint4 signFieldDim;
};

Texture3D<float> signFieldSRV : register(t0);
Texture3D<float4> normalFieldSRV : register(t1);

RWTexture3D<float> signFieldUAV : register(u0);

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)]
void smoothCS(uint3 tidx : SV_DispatchThreadID)
{
	const float BCVal = 10000.f;

	// diffuse sign field
	float signVal = 0.25f * signFieldSRV[tidx.xyz];
	float4 normal = normalFieldSRV[tidx.xyz];
	int3 idx;

	if (normal.w < 0.5f)
	{
		// pos x neighbor
		idx = tidx.xyz + int3(+1, 0, 0);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal -= normal.w * BCVal * normal.x;

		// neg x neighbor
		idx = tidx.xyz + int3(-1, 0, 0);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal += normal.w * BCVal * normal.x;

		// pos y neighbor
		idx = tidx.xyz + int3(0, +1, 0);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal -= normal.w * BCVal * normal.y;

		// neg y neighbor
		idx = tidx.xyz + int3(0, -1, 0);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal += normal.w * BCVal * normal.y;

		// pos z neighbor
		idx = tidx.xyz + int3(0, 0, +1);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal -= normal.w * BCVal * normal.z;

		// neg z neighbor
		idx = tidx.xyz + int3(0, 0, -1);
		normal = normalFieldSRV[idx];
		signVal += signFieldSRV[idx];
		signVal += normal.w * BCVal * normal.z;

		signVal *= (1.f / 6.25f);
	}
	else
	{
		signVal = 0.f;
	}

	if (tidx.x == 0) signVal = BCVal;
	if (tidx.x == signFieldDim.x - 1) signVal = BCVal;
	if (tidx.y == 0) signVal = BCVal;
	if (tidx.y == signFieldDim.y - 1) signVal = BCVal;
	if (tidx.z == 0) signVal = BCVal;
	if (tidx.z == signFieldDim.z - 1) signVal = BCVal;

	signFieldUAV[tidx.xyz] = signVal;
}