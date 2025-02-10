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
};

Buffer<float4> summarySRV : register(t0);

float4 gridSummaryDebugVisVS(float4 pos : POSITION, uint instanceIDIn : SV_InstanceID, uint vertexID : SV_VertexID) : SV_POSITION
{
	uint instanceID = instanceIDIn;

	float4 position = pos;

	{
		float4 worldLocation = summarySRV[4 * instanceID + 0];
		float4 worldHalfSize = summarySRV[4 * instanceID + 1];
		float4 velocity		 = summarySRV[4 * instanceID + 2];
		float4 density		 = summarySRV[4 * instanceID + 3];

		float3 dir = 0.f.xxx;
		if (velocity.w > 0.f)
		{
			dir = velocity.xyz / velocity.w;
		}
		if (velocity.w < 1.f)
		{
			dir = velocity.xyz;
		}

		if (vertexID == 0) position = float4(0.f, 0.f, 0.f, 1.f);
		if (vertexID == 1) position = float4(dir.x, dir.y, dir.z, 1.f);

		position = worldHalfSize * position + worldLocation;
	}
	position = mul(position, modelViewProj);

	return position;
}