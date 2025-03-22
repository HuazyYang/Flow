
cbuffer params : register(b0)
{
	float4x4 modelViewProj;
	float4x4 modelViewProjInv;
	float4 viewportInvScale;
	float4 viewportInvOffset;
	float4 dimInv;

	float4 minCoord;
	float4 maxCoord;
	float4 rayOrigin;
	float4 rayForwardDir;

	uint4 renderMode;
	float4 alphaScale;
};

struct Input
{
	float4 position : SV_POSITION;
	noperspective float3 rayDir : RAY_DIR;
};

SamplerState clampSampler : register(s4);

Texture3D<float4> densitySRV : register(t0);

// From GFSDK_VolumeRendering
bool intersectBox(float3 rayOrigin, float3 rayDir, float3 boxMin, float3 boxMax, out float tnear, out float tfar)
{
	// compute intersection of ray will all six bbox planes
	float3 invR = float3(1.f, 1.f, 1.f) / rayDir;
	float3 tbot = (boxMin - rayOrigin) * invR;
	float3 ttop = (boxMax - rayOrigin) * invR;

	// re-order intersections to find smallest and largest on each axis
	float3 tmin = min(ttop, tbot);
	float3 tmax = max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	tnear = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
	tfar = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

	return tfar > tnear;
}

float4 volumeRenderBoxPS(Input input) : SV_TARGET
{
	float tmin, tmax;

	const float3 ep3 = 0.0001f.xxx;

	float3 bboxMin = minCoord.xyz - ep3;
	float3 bboxMax = maxCoord.xyz + ep3;

	bool hit = intersectBox(rayOrigin.xyz, input.rayDir, bboxMin, bboxMax, tmin, tmax);

	// ensure traversal of ray only (not the whole line)
	tmin = max(0.f, tmin);

	float4 sum = float4(0.f, 0.f, 0.f, 1.f);

	if (hit)
	{
		tmin = round(tmin);
		tmax = round(tmax);

		int numSteps = int(tmax - tmin);

		float3 coord = tmin * input.rayDir + rayOrigin.xyz;

		float3 uvw = coord * dimInv.xyz;
		float3 uvwStep = input.rayDir * dimInv.xyz;

		for (int i = 0; i < numSteps; i++)
		{
			float4 normal = densitySRV.SampleLevel(clampSampler, uvw, 0);

			float4 color = 0.f.xxxx;
			if (renderMode.x == 0)
			{
				color.r = saturate(normal.x) + 0.5f * (saturate(-normal.y) + saturate(-normal.z));
				color.g = saturate(normal.y) + 0.5f * (saturate(-normal.x) + saturate(-normal.z));
				color.b = saturate(normal.z) + 0.5f * (saturate(-normal.x) + saturate(-normal.y));
				color.a = (1.f / 3.f) * (abs(normal.x) + abs(normal.y) + abs(normal.z));
			}
			if (renderMode.x == 1)
			{
				if (normal.x > 0.01f)
				{
					if (normal.x < 1.f) color = float4(0.f, 0.0f, 0.75f, 0.1f);
					if (normal.x < 0.5f) color = float4(0.5f, 0.0f, 0.f, 0.5f);
					if (normal.x < 0.25f) color = float4(0.f, 0.5f, 0.f, 0.5f);
				}

				if (normal.x < 0.f) color = float4(0.75f, 0.75f, 0.75f, 1.f);
			}
			if (renderMode.x == 2)
			{
				color.ba += 0.1f;
			}

			color = saturate(color);

			// scale alpha
			color.a *= alphaScale.w;
			// blend
			sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
			sum.a = (1.f - color.a) * sum.a;
			// advance uvw
			uvw += uvwStep;
		}
	}

	return sum;
}