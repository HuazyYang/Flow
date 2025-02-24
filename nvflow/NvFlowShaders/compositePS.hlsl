
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

float4 compositePS(Input input) : SV_TARGET
{
	float4 color = textureSRV.SampleLevel(clampSampler,input.uv,0);

	//int2 uv = int2(64.f * input.uv);
	//if ((uv.x ^ uv.y) & 1) color.rb += 0.5f;

	//if (all(abs(input.uv - 0.5f) < 0.01f))
	//{
	//	color.g += 0.5f;
	//}

	return color;
}