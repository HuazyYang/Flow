
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

float4 compositePS_LMS(Input input) : SV_TARGET
{
	float2 inuv = screenPercent.xy * uvDistort(screenPercent.zw * input.uv);

	float4 color = textureSRV.SampleLevel(borderSampler, inuv, 0);

	//int2 uv = int2(64.f * inuv);
	//if ((uv.x ^ uv.y) & 1) color.rb += 0.5f;

	//if (all(abs(inuv - 0.5f) < 0.01f))
	//{
	//	color.g += 0.5f;
	//}

	return color;
}