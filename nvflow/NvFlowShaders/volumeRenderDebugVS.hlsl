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
};

Buffer<uint> sortedBlockListSRV : register(t0);

float4 volumeRenderDebugVS( float4 pos : POSITION, uint instanceIDIn : SV_InstanceID) : SV_POSITION
{
	uint instanceID = instanceIDIn - 1;

	float4 position = pos;

	if (instanceIDIn > 0)
	{
		int3 vBlockIdx = tableVal_to_coord(sortedBlockListSRV[instanceID]);

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
	}
	position = mul(position, modelViewProj);

	return position;
}