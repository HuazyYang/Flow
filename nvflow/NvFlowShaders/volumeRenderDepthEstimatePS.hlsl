/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
#include "../NvFlow/volumeRenderShaderParams.h"
};

struct Input
{
	float4 position : SV_POSITION;
	noperspective float3 rayDirVirtual : RAY_DIR;
	float3 minVirtual : MIN_VIRTUAL;
	float3 maxVirtual : MAX_VIRTUAL;
	uint3 vBlockIdx : VBLOCK_IDX;
	uint3 rBlockIdx_layer0 : RBLOCK_IDX_LAYER0;
};

//Texture2D<float> depthSRV : register(t0);
//Texture2D<float> depthMinSRV : register(t1);

Texture3D<float4> densitySRV_layer0 : register(t2);
Texture1D<float4> colorMapSRV_layer0 : register(t3);

// From GFSDK_VolumeRendering
bool intersectBox(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax, out float tnear, out float tfar)
{
	// compute intersection of ray will all six bbox planes
	float3 invR = float3(1.f, 1.f, 1.f) / rayDir;
	float3 tbot = (boxMin - rayOrigin) * invR;
	float3 ttop = (boxMax - rayOrigin) * invR;

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = min(ttop, tbot);
	float3 tmax = max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	tnear = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
	tfar = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

	return tfar > tnear;
}

VIRTUAL_TO_REAL_LINEAR2(VirtualToReal, valueParams);

#define RAY_MARCH_SAMPLE 1

#define LAYER_NAME layer0
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME

float4 volumeRenderDepthEstimatePS(Input input) : SV_TARGET
{
	float tmin, tmax;

	const float3 ep3 = 0.0001f.xxx;

	float3 bboxMin = input.minVirtual - ep3;
	float3 bboxMax = input.maxVirtual + ep3;

	bool hit = intersectBox(rayOriginVirtual.xyz, input.rayDirVirtual, bboxMin, bboxMax, tmin, tmax);

	// nice for debugging depth
	//if (tdepth < tmax)
	//{
	//	return float4(0.f, 1.f, 0.f, 0.5f);
	//}

	// ensure traversal of ray only (not the whole line)
	tmin = max(0.f, tmin);

	float4 massCount = 0.f.xxxx;

	if (hit)
	{
		tmin = round(tmin);
		tmax = round(tmax);

		//// nice for debugging depth
		//if (depthHit)
		//{
		//	return float4(0.f, 1.f, 0.f, 0.25f);
		//}

		int numSteps = int(tmax - tmin);

		// vary tmin to reduce aliasing
		int parity = (int(input.position.x) & 1) ^ (int(input.position.y) & 1);
		//tmin += (parity == 0) ? 0.25f : -0.25f;
		tmin += (parity == 0) ? 0.125f : -0.125f;

		float3 vidx = tmin * input.rayDirVirtual + rayOriginVirtual.xyz;

		float3 uvw_layer0 = VirtualToReal(vidx, input.rBlockIdx_layer0, input.vBlockIdx) * valueParams.dimInv.xyz;
		float3 uvwStep = input.rayDirVirtual * valueParams.dimInv.xyz;

		for (int i = 0; i < numSteps;)
		{
			float4 sum = float4(0.f, 0.f, 0.f, 1.f);
			rayMarchSample_layer0(sum, uvw_layer0, tmin, i, alphaScale_layer0);

			massCount.x += (1.f - sum.w) * (i + tmin);
			massCount.y += (1.f - sum.w);

			// advance uvw
			uvw_layer0 += 8.f * uvwStep;
			i+=8;
		}
	}

	return 0.1f * massCount;
}