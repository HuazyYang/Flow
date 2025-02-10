
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

Texture2D<float> depthMaxSRV : register(t0);

cbuffer params : register(b0)
{
	NvFlowFloat4 uvScale;
}

float4 multiResDepthUpsamplePS(Input input) : SV_TARGET
{
	float4 depthMax4 = depthMaxSRV.Gather(clampSampler, input.uv);

	bool fault =
		(int(depthMax4.x) != int(depthMax4.y)) ||
		(int(depthMax4.x) != int(depthMax4.z)) ||
		(int(depthMax4.x) != int(depthMax4.w)) ;

	float depth = 0.f;
	if (!fault)
	{
		depth = depthMaxSRV.SampleLevel(clampPointSampler, input.uv, 0);
	}

	return depth;
}