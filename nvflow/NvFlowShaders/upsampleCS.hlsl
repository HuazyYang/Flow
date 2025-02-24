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

Texture3D<float> coarseSignFieldSRV : register(t0);

RWTexture3D<float> signFieldUAV : register(u0);

[numthreads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM)]
void upsampleCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 c = tidx.xyz - int3(1, 1, 1);
	int3 c000 = c >> 1;

	float fx = bool(c.x & 1) ? 0.75f : 0.25f;
	float fy = bool(c.y & 1) ? 0.75f : 0.25f;
	float fz = bool(c.z & 1) ? 0.75f : 0.25f;
	float ofx = 1.f - fx;
	float ofy = 1.f - fy;
	float ofz = 1.f - fz;

	float d000 = coarseSignFieldSRV[c000 + int3(0, 0, 0)];
	float d100 = coarseSignFieldSRV[c000 + int3(1, 0, 0)];
	float d010 = coarseSignFieldSRV[c000 + int3(0, 1, 0)];
	float d110 = coarseSignFieldSRV[c000 + int3(1, 1, 0)];
	float d001 = coarseSignFieldSRV[c000 + int3(0, 0, 1)];
	float d101 = coarseSignFieldSRV[c000 + int3(1, 0, 1)];
	float d011 = coarseSignFieldSRV[c000 + int3(0, 1, 1)];
	float d111 = coarseSignFieldSRV[c000 + int3(1, 1, 1)];

	float d = ofz*(ofy*(ofx*d000 + fx*d100) + fy*(ofx*d010 + fx*d110)) +
		fz*(ofy*(ofx*d001 + fx*d101) + fy*(ofx*d011 + fx*d111));

	signFieldUAV[tidx.xyz] = d;
}