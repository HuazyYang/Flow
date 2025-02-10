
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

cbuffer params : register(b0)
{
	float4 uvScale;
}

Texture2D<float4> colorSRV : register(t0);
Texture2D<float> depthMaxSRV : register(t1);

float4 multiResColorUpsamplePS(Input input) : SV_TARGET
{
	float4 depthMax4 = depthMaxSRV.Gather(clampSampler, input.uv);

	bool fault =
		(int(depthMax4.x) != int(depthMax4.y)) ||
		(int(depthMax4.x) != int(depthMax4.z)) ||
		(int(depthMax4.x) != int(depthMax4.w));

	float4 sum = float4(0.f, 0.f, 0.f, 1.f);

	if (!fault)
	{
		sum = colorSRV.SampleLevel(clampSampler, input.uv, 0);;
	}

	return sum;
}