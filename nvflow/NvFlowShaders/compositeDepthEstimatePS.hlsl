
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

struct PSOutput
{
	//float4 color : SV_TARGET;
	float depth : SV_DEPTH;
};

PSOutput compositeDepthEstimatePS(Input input)
{
	PSOutput output;

	float4 compositeColor = textureSRV.SampleLevel(clampSampler,input.uv,0);
	float4 depthEstimate = depthEstimateSRV.SampleLevel(borderSampler, input.uv, 0);

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