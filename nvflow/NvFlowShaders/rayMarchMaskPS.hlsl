
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

Texture2D<float> depthMinSRV : register(t0);
Texture2D<float> depthMaskSRV : register(t1);

cbuffer params : register(b0)
{
	NvFlowFloat4 uvScale;

	NvFlowFloat4 depthInvTransform;
}

void rayMarchMaskPS(Input input)
{
	float depthMin = depthMinSRV.SampleLevel(clampPointSampler, input.uv, 0);
	float depthMask = depthMaskSRV.SampleLevel(clampPointSampler, input.uv, 0);

	depthMask =
		(depthMask * depthInvTransform.x + depthInvTransform.y) /
		(depthMask * depthInvTransform.z + depthInvTransform.w);

	if (depthMin < depthMask)
	{
		discard;
	}
}