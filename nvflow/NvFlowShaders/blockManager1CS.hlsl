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

RWTexture3D<uint> userMappingUAV : register(u0);

Texture3D<uint> fieldMappingSRV : register(t0);
Texture3D<float> velocitySummarySRV : register(t1);
Texture3D<float4> densitySummarySRV : register(t2);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void blockManager1CS(uint3 mapIdx : SV_DispatchThreadID)
{
	uint userVal = userMappingUAV[mapIdx];
	uint fieldVal = fieldMappingSRV[mapIdx];

	float vel2 = 0.f;
	float4 den2_4 = 0.f.xxxx;

	// summary is only valid if region active when summary was generated
	if (fieldVal != 0u)
	{
		vel2 = velocitySummarySRV[mapIdx >> velFactorBits.xyz];
		den2_4 = densitySummarySRV[mapIdx >> denFactorBits.xyz];
	}
	
	float velocity = sqrt(vel2);
	float density = sqrt(den2_4.w);
	float temp = sqrt(den2_4.x);
	float fuel = sqrt(den2_4.y);

	if (userVal != 0)
	{
		fieldVal |= 0x01;
	}
	else
	{
		fieldVal &= ~0x01;
	}

	float importance =
		velocityWeight * max(0.f, velocity - velocityThreshold) +
		smokeWeight * max(0.f, density - smokeThreshold) +
		tempWeight * max(0.f, temp - tempThreshold) +
		fuelWeight * max(0.f, fuel - fuelThreshold);

	importance = max(0.f, importance - importanceThreshold);

	if (importance > 0.f)
	{
		fieldVal |= 0x04;
	}
	else
	{
		fieldVal &= ~0x04;
	}

	userMappingUAV[mapIdx] = fieldVal;
}