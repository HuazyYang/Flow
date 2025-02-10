
#ifndef DEPTH_MASK
#define DEPTH_MASK 0
#endif

struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

cbuffer params : register(b0)
{
#include "../NvFlow/depthDownsampleShaderParams.h"
}

Texture2D<float> textureSRV : register(t0);
#if DEPTH_MASK
Texture2D<float> depthMaskSRV : register(t1);
#endif

float depthDownsamplePS(Input input) : SV_TARGET
{
	float4 depth4 = textureSRV.Gather(borderSampler,input.uv,0);
	float depthVal;
	if (depth_reverseZ.x == 0u)
	{
		depthVal = max(max(depth4.x, depth4.y), max(depth4.z, depth4.w));
	}
	if (depth_reverseZ.x != 0u)
	{
		depthVal = min(min(depth4.x, depth4.y), min(depth4.z, depth4.w));
	}

	//float depthVal = textureSRV.SampleLevel(borderSampler,input.uv,0);

#if DEPTH_MASK
	// depth mask override
	{
		float2 maskUV = depth_uvScale.xy * (viewportInvScale.xy * input.position.xy + viewportInvOffset.xy) + depth_uvScale.zw;
		float maskVal = depthMaskSRV.SampleLevel(borderSampler, maskUV, 0);

		if (depth_reverseZ.x == 0u)
		{
			if (maskVal < depthVal)
			{
				//depthVal = (maskVal > 0.f) ? 1.f : 0.f;
				depthVal = 1.f;
			}
		}
		if (depth_reverseZ.x != 0u)
		{
			if (maskVal > depthVal)
			{
				depthVal = 0.f;
			}
		}
	}
#endif

	return depthVal;
}