
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "framework.hlsli"

Texture2D<float> depthSRV : register(t0);

cbuffer params : register(b0)
{
	float4 uvScale;
	float4 depthInvTransform;
	float tmax;
	float tbias;
	float pad0;
	float pad1;
}

float4 multiResDepthDownsamplePS(Input input) : SV_TARGET
{
	float4 depth4 = depthSRV.Gather(clampSampler, input.uv);

	float4 t4 =
		(depth4 * depthInvTransform.x + depthInvTransform.y) /
		(depth4 * depthInvTransform.z + depthInvTransform.w);

	float tmin = min(min(t4.x, t4.y), min(t4.z, t4.w));

	// add bias to create layer in front of depth
	tmin -= tbias;
	tmin = max(0.f, tmin);

	return floor(min(tmax, tmin));
}