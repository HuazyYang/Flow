/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "framework.hlsli"

cbuffer params : register(b0)
{
	NvFlowFloat4x4 modelViewProj;
	NvFlowFloat4 vGridDimInv;
	NvFlowUint4 vGridDim;
	NvFlowFloat4 linearDepthTransform;
};

//Buffer<uint> sortedBlockListSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t0);

float4 depthLinear4(float3 ndc)
{
	float a = linearDepthTransform.z * ndc.z + linearDepthTransform.w;
	float b = linearDepthTransform.y * ndc.z;

	return float4(a * ndc.xy, b, a);
}

float4 volumeShadowDebugVS(float4 pos : POSITION, uint instanceIDIn : SV_InstanceID) : SV_POSITION
{
	uint instanceID = instanceIDIn - 1;

	float4 position = pos;

	if (instanceIDIn > 0)
	{
		//int3 vBlockIdx = tableVal_to_coord(sortedBlockListSRV[instanceID]);
		int3 vBlockIdx = int3(
			instanceID % vGridDim.x,
			(instanceID / vGridDim.x) % vGridDim.y,
			instanceID / (vGridDim.x * vGridDim.y)
			);

		uint blockTableVal = blockTableSRV[vBlockIdx];

		if (blockTableVal != 0u)
		{
			// transform the unit cube for this block
			// This is the block half size when the virtual space maps [-1.f,+1.f]
			float4 blockHalfSize = float4(
				vGridDimInv.x,
				vGridDimInv.y,
				vGridDimInv.z,
				1.f
				);

			float4 blockLocation = float4(
				2.f * (float(vBlockIdx.x) + 0.5f) * vGridDimInv.x - 1.f,
				2.f * (float(vBlockIdx.y) + 0.5f) * vGridDimInv.y - 1.f,
				2.f * (float(vBlockIdx.z) + 0.5f) * vGridDimInv.z - 1.f,
				0.f
				);

			position = blockHalfSize * position + blockLocation;

			// for the shadow case, z is range [0,1] instead of [-1,+1]
			position.z = 0.5f * position.z + 0.5f;

			position = depthLinear4(position.xyz);
		}
		else
		{
			position = float4(1.f, 1.f, 1.f, 0.f);
		}
	}
	else
	{
		// for the shadow case, z is range [0,1] instead of [-1,+1]
		position.z = 0.5f * position.z + 0.5f;
	}
	position = mul(position, modelViewProj);

	return position;
}