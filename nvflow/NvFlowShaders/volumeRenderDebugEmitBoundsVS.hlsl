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

Buffer<float4> boundsListSRV : register(t0);

float4 volumeRenderDebugEmitBoundsVS(float4 pos : POSITION, uint instanceID : SV_InstanceID) : SV_POSITION
{
	float4 position = pos;

	float4x4 bounds = float4x4(
		boundsListSRV[4 * instanceID + 0],
		boundsListSRV[4 * instanceID + 1],
		boundsListSRV[4 * instanceID + 2],
		boundsListSRV[4 * instanceID + 3]
		);
	
	position = mul(position, bounds);
	position = mul(position, modelViewProj);

	return position;
}