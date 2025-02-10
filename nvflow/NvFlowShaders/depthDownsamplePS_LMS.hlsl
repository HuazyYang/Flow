
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

float2 uvDistort(float2 uv, out float depthScale)
{
	float A, B;
	float2 scaleBiasX, scaleBiasY;

	float4 clipPos = float4(2.f * uv.xy - 1.f, 0.f, 1.f);

	if (clipPos.x < 0.f)
	{
		A = -warpLeft;
		scaleBiasX = leftX;
	}
	else
	{
		A = +warpRight;
		scaleBiasX = rightX;
	}

	if (clipPos.y < 0.f)
	{
		B = -warpUp;
		scaleBiasY = upY;
	}
	else
	{
		B = +warpDown;
		scaleBiasY = downY;
	}

	float4 warpedPos = clipPos;
	warpedPos.w = clipPos.w + clipPos.x * A + clipPos.y * B;

	depthScale = warpedPos.w;

	warpedPos.xyz /= warpedPos.w;
	warpedPos.w = 1.f;

	float2 ret = warpedPos.xy;
	ret.x = ret.x * scaleBiasX.x + scaleBiasX.y;
	ret.y = ret.y * scaleBiasY.x + scaleBiasY.y;

	return ret;
}

float depthDownsamplePS_LMS(Input input) : SV_TARGET
{
	float depthScale;
	float2 inuv = uvDistort(input.uv, depthScale);

	float4 depth4 = textureSRV.Gather(borderSampler, inuv, 0);
	float depthVal;
	if (depth_reverseZ.x == 0u)
	{
		depthVal = max(max(depth4.x, depth4.y), max(depth4.z, depth4.w));
	}
	if (depth_reverseZ.x != 0u)
	{
		depthVal = min(min(depth4.x, depth4.y), min(depth4.z, depth4.w));
	}

	// need depth scaling with LMS
	depthVal *= depthScale;

	//float depthVal = depthScale * textureSRV.SampleLevel(borderSampler, inuv, 0);

	depthVal = saturate(depthVal);

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