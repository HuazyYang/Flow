
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
Texture2D<float> depthSRV : register(t1);

Texture2D<float> sceneDepthSRV : register(t2);

Texture2D<float> depthMaskSRV : register(t3);

float transformInvZ(float z)
{
	return
		(z * depthInvTransform.x + depthInvTransform.y) *
		rcp(z * depthInvTransform.z + depthInvTransform.w);
}

float4 transformInvZ4(float4 z4)
{
	return
		(z4 * depthInvTransform.x + depthInvTransform.y) *
		rcp(z4 * depthInvTransform.z + depthInvTransform.w);
}

float4 compositeSmoothPS(Input input) : SV_TARGET
{
	float2 inuv = input.uv;

	float4 color = textureSRV.SampleLevel(clampSampler, inuv,0);

	float2 sceneUV = scene_uvScale.xy * input.position.xy + scene_uvScale.zw;

	// fetch sceneDepth and transform to T space
	float sceneDepth = sceneDepthSRV.SampleLevel(clampPointSampler, sceneUV, 0);
	sceneDepth = transformInvZ(sceneDepth);

	float maskDepth = depthMaskSRV.SampleLevel(clampSampler, inuv, 0);
	maskDepth = transformInvZ(maskDepth);

	// limit maximum T depth
	sceneDepth = min(tlimitRange.y, sceneDepth);

	if (maskDepth < sceneDepth)
	{
		sceneDepth = tlimitRange.y;
	}

	float2 texCoord = inuv * uvStepSize.zw;
	float2 texCoordi = round(texCoord);
	float2 st = texCoord - texCoordi + 0.5f.xx;
	float2 fetchUV = (texCoordi)* uvStepSize.xy;

	float4 w4 = float4(
		(1.f - st.x) * (0.f + st.y),
		(0.f + st.x) * (0.f + st.y),
		(0.f + st.x) * (1.f - st.y),
		(1.f - st.x) * (1.f - st.y)
		);

	// gather offscreen depth and transform to T space
	float4 depth4 = depthSRV.Gather(clampSampler, fetchUV, 0);
	depth4 = transformInvZ4(depth4);

	// limit maximum T depth
	depth4 = min(tlimitRange.y, depth4);

	// interpolate T space depth
	float depth = dot(w4, depth4);

	float err = abs(depth - sceneDepth);

	if (err > 16.f)
	{
		[unroll]
		for (int passID = 0; passID < 3; passID++)
		{
			float4 dw4 = w4 * depth4;

			float4 newDepth4 = (depth - dw4) * rcp(1.f - w4);

			float4 newErr4 = abs(newDepth4 - sceneDepth.xxxx);

			// find the minimum error
			float minErr = min(err, min(min(newErr4.x, newErr4.y), min(newErr4.z, newErr4.w)));

			if (minErr == newErr4.x) w4.x = 0.f;
			else if (minErr == newErr4.y) w4.y = 0.f;
			else if (minErr == newErr4.z) w4.z = 0.f;
			else if (minErr == newErr4.w) w4.w = 0.f;

			// normalize new weights
			float wnorm = rcp(w4.x + w4.y + w4.z + w4.w);
			w4 *= wnorm;
		}

		float4 r4 = textureSRV.GatherRed(clampPointSampler, fetchUV, 0);
		float4 g4 = textureSRV.GatherGreen(clampPointSampler, fetchUV, 0);
		float4 b4 = textureSRV.GatherBlue(clampPointSampler, fetchUV, 0);
		float4 a4 = textureSRV.GatherAlpha(clampPointSampler, fetchUV, 0);

		float4 c;
		c.r = dot(w4, r4);
		c.g = dot(w4, g4);
		c.b = dot(w4, b4);
		c.a = dot(w4, a4);

		color = c;

		//color.g = saturate(err);
		//color.b = 1.f - color.g;
	}

	return color;
}