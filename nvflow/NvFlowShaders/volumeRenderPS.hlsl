/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

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
#if (NUM_LAYERS > 1)
	uint3 rBlockIdx_layer1 : RBLOCK_IDX_LAYER1;
#endif
#if (NUM_LAYERS > 2)
	uint3 rBlockIdx_layer2 : RBLOCK_IDX_LAYER2;
#endif
#if (NUM_LAYERS > 3)
	uint3 rBlockIdx_layer3 : RBLOCK_IDX_LAYER3;
#endif
};

Texture2D<float> depthSRV : register(t0);
Texture2D<float> depthMinSRV : register(t1);

Texture3D<float4> densitySRV_layer0 : register(t2);
Texture1D<float4> colorMapSRV_layer0 : register(t3);
#if (NUM_LAYERS > 1)
Texture3D<float4> densitySRV_layer1 : register(t4);
Texture1D<float4> colorMapSRV_layer1 : register(t5);
#endif
#if (NUM_LAYERS > 2)
Texture3D<float4> densitySRV_layer2 : register(t6);
Texture1D<float4> colorMapSRV_layer2 : register(t7);
#endif
#if (NUM_LAYERS > 3)
Texture3D<float4> densitySRV_layer3 : register(t8);
Texture1D<float4> colorMapSRV_layer3 : register(t9);
#endif

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

void computeDepth_trange(out float tdepthMin, out float tdepthMax, float4 position, float3 rayDirVirtual)
{
	// model space location of depth value
	float2 depthFetchCoord = depthUVTransform.xy * (viewportInvScale.xy * position.xy + viewportInvOffset.xy) + depthUVTransform.zw;
	//float depthValue = depthSRV[2 * int2(input.position.xy)];
	float depthValue = depthSRV.SampleLevel(clampPointSampler, depthFetchCoord, 0.f);
	tdepthMax =
		(depthValue * depthMaxInvTransform.x + depthMaxInvTransform.y) /
		(depthValue * depthMaxInvTransform.z + depthMaxInvTransform.w);

	tdepthMin = 0.f;

	if (depthMinInvTransform.y != 0.f || depthMinInvTransform.x != 0.f)
	{
		depthValue = depthMinSRV.SampleLevel(clampPointSampler, depthFetchCoord, 0.f);

		tdepthMin =
			(depthValue * depthMinInvTransform.x + depthMinInvTransform.y) /
			(depthValue * depthMinInvTransform.z + depthMinInvTransform.w);
	}
}

VIRTUAL_TO_REAL_LINEAR2(VirtualToReal, valueParams);

#define RAY_MARCH_SAMPLE 1

#define LAYER_NAME layer0
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME
#if (NUM_LAYERS > 1)
#define LAYER_NAME layer1
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME
#endif
#if (NUM_LAYERS > 2)
#define LAYER_NAME layer2
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME
#endif
#if (NUM_LAYERS > 3)
#define LAYER_NAME layer3
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME
#endif

float4 volumeRenderPS(Input input) : SV_TARGET
{
	float tmin, tmax;
	float tdepthMin, tdepthMax;

	const float3 ep3 = 0.0001f.xxx;

	float3 bboxMin = input.minVirtual - ep3;
	float3 bboxMax = input.maxVirtual + ep3;

	bool hit = intersectBox(rayOriginVirtual.xyz, input.rayDirVirtual, bboxMin, bboxMax, tmin, tmax);

	computeDepth_trange(tdepthMin, tdepthMax, input.position, input.rayDirVirtual);

	// distance from step center to step end
	tdepthMax += 0.5f;

	// nice for debugging depth
	//if (tdepth < tmax)
	//{
	//	return float4(0.f, 1.f, 0.f, 0.5f);
	//}

	// ensure traversal of ray only (not the whole line)
	tmin = max(0.f, tmin);

	float4 sum = float4(0.f,0.f,0.f,1.f);

	if (hit)
	{
		float tdepthMaxf = tdepthMax;

		tmin = round(tmin);
		tmax = round(tmax);
		tdepthMax = round(tdepthMax);
		tdepthMin = round(tdepthMin);

		// limit tmin based on tdepthMin
		tmin = max(tdepthMin, tmin);

		bool depthHit = (int(tdepthMax) > int(tmin)) && (int(tdepthMax) <= int(tmax));

		//// nice for debugging depth
		//if (depthHit)
		//{
		//	return float4(0.f, 1.f, 0.f, 0.25f);
		//}

		// limit tmax based on tdepthMax
		tmax = min(tdepthMax, tmax);

		int numSteps = int(tmax - tmin);
		if (depthHit) numSteps--;

		// vary tmin to reduce aliasing
		int parity = (int(input.position.x) & 1) ^ (int(input.position.y) & 1);
		tmin += (parity == 0) ? 0.5f : 0.f;

		float3 vidx = tmin * input.rayDirVirtual + rayOriginVirtual.xyz;

		float3 uvw_layer0 = VirtualToReal(vidx, input.rBlockIdx_layer0, input.vBlockIdx) * valueParams.dimInv.xyz;
		#if (NUM_LAYERS > 1)
		float3 uvw_layer1 = VirtualToReal(vidx, input.rBlockIdx_layer1, input.vBlockIdx) * valueParams.dimInv.xyz;
		#endif
		#if (NUM_LAYERS > 2)
		float3 uvw_layer2 = VirtualToReal(vidx, input.rBlockIdx_layer2, input.vBlockIdx) * valueParams.dimInv.xyz;
		#endif
		#if (NUM_LAYERS > 3)
		float3 uvw_layer3 = VirtualToReal(vidx, input.rBlockIdx_layer3, input.vBlockIdx) * valueParams.dimInv.xyz;
		#endif
		float3 uvwStep = input.rayDirVirtual * valueParams.dimInv.xyz;

		for (int i = 0; i < numSteps; i++)
		{
			rayMarchSample_layer0(sum, uvw_layer0, tmin, i, alphaScale_layer0);
			#if (NUM_LAYERS > 1)
			rayMarchSample_layer1(sum, uvw_layer1, tmin, i, alphaScale_layer1);
			#endif
			#if (NUM_LAYERS > 2)
			rayMarchSample_layer2(sum, uvw_layer2, tmin, i, alphaScale_layer2);
			#endif
			#if (NUM_LAYERS > 3)
			rayMarchSample_layer3(sum, uvw_layer3, tmin, i, alphaScale_layer3);
			#endif
			// advance uvw
			uvw_layer0 += uvwStep;
			#if (NUM_LAYERS > 1)
			uvw_layer1 += uvwStep;
			#endif
			#if (NUM_LAYERS > 2)
			uvw_layer2 += uvwStep;
			#endif
			#if (NUM_LAYERS > 3)
			uvw_layer3 += uvwStep;
			#endif
		}

		// improved depth test
		if (depthHit)
		{
			// do an extra sample
			float w = (tdepthMaxf - tdepthMax + 0.5f);
			rayMarchSample_layer0(sum, uvw_layer0, tmin, i, alphaScale_layer0 * w);
			#if (NUM_LAYERS > 1)
			rayMarchSample_layer1(sum, uvw_layer1, tmin, i, alphaScale_layer1 * w);
			#endif
			#if (NUM_LAYERS > 2)
			rayMarchSample_layer2(sum, uvw_layer2, tmin, i, alphaScale_layer2 * w);
			#endif
			#if (NUM_LAYERS > 3)
			rayMarchSample_layer3(sum, uvw_layer3, tmin, i, alphaScale_layer3 * w);
			#endif
		}
	}

	return sum;
}