
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

cbuffer params : register(b0)
{
#include "../NvFlow/compositeShaderParams.h"
}

Texture2D<float4> textureSRV : register(t0);
Texture2D<float4> depthEstimateSRV : register(t1);

float2 uvDistort(float2 uv)
{
	float A, B;
	float2 scaleBiasX, scaleBiasY;

	if (uv.x < leftX.y)
	{
		A = +warpLeft;
		scaleBiasX = leftX;
	}
	else
	{
		A = -warpRight;
		scaleBiasX = rightX;
	}

	if (uv.y < upY.y)
	{
		B = +warpUp;
		scaleBiasY = upY;
	}
	else
	{
		B = -warpDown;
		scaleBiasY = downY;
	}

	float2 uv1;
	uv1.x = (uv.x - scaleBiasX.y) * scaleBiasX.x;
	uv1.y = (uv.y - scaleBiasY.y) * scaleBiasY.x;
	float4 clipPos = float4(uv1.xy, 0.f, 1.f);

	float4 warpedPos = clipPos;
	warpedPos.w = clipPos.w + clipPos.x * A + clipPos.y * B;
	warpedPos.xyz /= warpedPos.w;
	warpedPos.w = 1.f;

	float2 ret = warpedPos.xy;
	ret.x = ret.x * 0.5f + 0.5f;
	ret.y = ret.y * 0.5f + 0.5f;

	return ret;
}

struct PSOutput
{
	//float4 color : SV_TARGET;
	float depth : SV_DEPTH;
};

PSOutput compositeDepthEstimatePS_LMS(Input input)
{
	PSOutput output;

	float2 inuv = screenPercent.xy * uvDistort(screenPercent.zw * input.uv);

	float4 compositeColor = textureSRV.SampleLevel(borderSampler, inuv, 0);
	float4 depthEstimate = depthEstimateSRV.SampleLevel(borderSampler, inuv, 0);

	float4 color = float4(0.f, 0.f, 0.f, 1.f);
	if (compositeMode.x == 1u)
	{
		color = depthEstimate;
	}

	if (depthEstimate.y == 0.f)
	{
		discard;
	}

	float alpha = 1.f - saturate(compositeColor.a);
	float intensity = max(compositeColor.r, max(compositeColor.g, compositeColor.b));

	if (alpha < depthAlphaThreshold && intensity < depthIntensityThreshold)
	{
		discard;
	}

	float t = depthEstimate.x / depthEstimate.y;
	float depth = (-t * depthInvTransform.w + depthInvTransform.y) /
		(t * depthInvTransform.z - depthInvTransform.x);

	//output.color = color;
	output.depth = depth;
	return output;
}