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
	NvFlowUint4 shapeType;
};

#define eNvFlowShapeTypeSphere 1
#define eNvFlowShapeTypeBox 2
#define eNvFlowShapeTypeCapsule 3

Buffer<float4> boundsListSRV : register(t0);

float4 volumeRenderDebugShapesSimpleVS(float4 pos : POSITION, uint instanceID : SV_InstanceID) : SV_POSITION
{
	float4 position = pos;

	float4x4 localToWorld = float4x4(
		boundsListSRV[5 * instanceID + 0],
		boundsListSRV[5 * instanceID + 1],
		boundsListSRV[5 * instanceID + 2],
		boundsListSRV[5 * instanceID + 3]
		);

	float4 shapeDesc = boundsListSRV[5 * instanceID + 4];

	// distort here based on shapeDesc and shapeType
	switch (shapeType.x)
	{
		case eNvFlowShapeTypeSphere:
		{
			position.xyz *= shapeDesc.x;
			break;
		}
		case eNvFlowShapeTypeCapsule:
		{
			// get sphere size correct first
			position.xyz *= shapeDesc.x;

			// add offsets on x to establish length
			float offset = 0.5f * shapeDesc.y - shapeDesc.x;

			if (position.x < 0.f) position.x -= offset;
			else position.x += offset;
			break;
		}
		case eNvFlowShapeTypeBox:
		{
			position.xyz *= shapeDesc.xyz;
			break;
		}
	}

	position = mul(position, localToWorld);
	position = mul(position, modelViewProj);

	return position;
}