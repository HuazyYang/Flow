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
#include "../NvFlow/volumeRenderShaderParams.h"
};

struct Output
{
	float4 position : SV_POSITION;
	noperspective float3 rayDirVirtual : RAY_DIR;
	float3 minVirtual : MIN_VIRTUAL;
	float3 maxVirtual : MAX_VIRTUAL;
	uint3 vBlockIdx : VBLOCK_IDX;
	uint3 rBlockIdx_layer0 : RBLOCK_IDX_LAYER0;
};

Buffer<uint> sortedBlockListSRV : register(t0);
Texture3D<uint> blockTableSRV_layer0 : register(t1);

Output volumeRenderDepthEstimateVS(float4 pos : POSITION, uint instanceID : SV_InstanceID)
{
	Output output;

	int3 vBlockIdx = tableVal_to_coord(sortedBlockListSRV[instanceID + blockListStartCount.x]);
	output.vBlockIdx = vBlockIdx;

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

	output.position = blockHalfSize * pos + blockLocation;
	output.position = mul(output.position, modelViewProj);

	// get real block location
	output.rBlockIdx_layer0 = tableVal_to_coord(blockTableSRV_layer0[vBlockIdx]);

	// compute min/max virtual bounds	
	output.minVirtual = float3(valueParams.blockDim.xyz * vBlockIdx);
	output.maxVirtual = float3(valueParams.blockDim.xyz * vBlockIdx + valueParams.blockDim.xyz);

	// compute ray direction in density texture uvw space for this this block
	float3 vidx = float3(
		(pos.x < 0.f) ? output.minVirtual.x : output.maxVirtual.x,
		(pos.y < 0.f) ? output.minVirtual.y : output.maxVirtual.y,
		(pos.z < 0.f) ? output.minVirtual.z : output.maxVirtual.z);

	float3 rayDir = vidx - rayOriginVirtual.xyz;
	output.rayDirVirtual = rayDir * dot(rayForwardDirVirtual.xyz, rayForwardDirVirtual.xyz) / dot(rayForwardDirVirtual.xyz, rayDir);

	return output;
}