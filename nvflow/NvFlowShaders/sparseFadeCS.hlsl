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
	NvFlowUint4 fadeFieldDim;
	NvFlowFloat4 fadeFieldDimInv;
	NvFlowFloat4 blockTableVelocityDim;
	NvFlowFloat4 blockTableDensityDim;
};

RWTexture3D<float> fadeFieldVelocityUAV : register(u0);
RWTexture3D<float> fadeFieldDensityUAV : register(u1);

Texture3D<uint> blockTableVelocitySRV : register(t0);
Texture3D<uint> blockTableDensitySRV : register(t1);

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void sparseFadeCS(uint3 fadeFieldIdx : SV_DispatchThreadID)
{
	float3 uvw = fadeFieldDimInv.xyz * (float3(fadeFieldIdx)+0.5f.xxx);

	uint blockTableVelocity = blockTableVelocitySRV[floor(blockTableVelocityDim.xyz * uvw)];
	uint blockTableDensity  = blockTableDensitySRV[floor(blockTableDensityDim.xyz * uvw)];

	bool enabledVelocity = (blockTableVelocity != 0u);
	bool enabledDensity = (blockTableVelocity != 0u) && (blockTableDensity != 0u);

	fadeFieldVelocityUAV[fadeFieldIdx] = enabledVelocity ? 1.f : 0.f;
	fadeFieldDensityUAV[fadeFieldIdx] = enabledDensity ? 1.f : 0.f;
}