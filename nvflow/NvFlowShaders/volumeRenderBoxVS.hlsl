/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

cbuffer params : register(b0)
{
	float4x4 modelViewProj;
	float4x4 modelViewProjInv;
	float4 viewportInvScale;
	float4 viewportInvOffset;
	float4 dimInv;

	float4 minCoord;
	float4 maxCoord;
	float4 rayOrigin;
	float4 rayForwardDir;

	uint4 renderMode;
	float4 alphaScale;
};

struct Output
{
	float4 position : SV_POSITION;
	noperspective float3 rayDir : RAY_DIR;
};

Output volumeRenderBoxVS(float4 pos : POSITION)
{
	Output output;

	output.position = mul(pos, modelViewProj);

	float3 coord = float3(
		(pos.x < 0.f) ? minCoord.x : maxCoord.x,
		(pos.y < 0.f) ? minCoord.y : maxCoord.y,
		(pos.z < 0.f) ? minCoord.z : maxCoord.z
		);

	float3 rayDir = coord - rayOrigin.xyz;
	output.rayDir = rayDir * dot(rayForwardDir.xyz, rayForwardDir.xyz) / dot(rayForwardDir.xyz, rayDir);

	return output;
}