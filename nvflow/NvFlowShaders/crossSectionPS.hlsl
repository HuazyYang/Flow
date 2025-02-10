
struct Input
{
	float4 position : SV_POSITION;
	float2 uv : UV;
};

#include "frameworkHybrid.hlsli"

cbuffer params : register(b0)
{
	#include "../NvFlow/crossSectionShaderParams.h"
}

Texture3D<float4> valueSRV : register(t0);
Texture3D<uint> blockTableSRV : register(t1);
Texture1D<float4> colorMapSRV_layer0 : register(t2);

VIRTUAL_TO_REAL_LINEAR(VirtualToReal, blockTableSRV, valueParams)

#define LAYER_NAME layer0
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME

float4 crossSectionPS(Input input) : SV_TARGET
{
	float3 vidxNorm = float3(input.uv, 0.5f);

	float3 ndc = 2.f * vidxNorm - 1.f;

	ndc = ndc - crossSectionPosition.xyz;

	ndc *= crossSectionScale.xyz;

	vidxNorm = 0.5f * ndc + 0.5f;

	float3 vidxf = valueParams.vdim.xyz * vidxNorm;

	if (crossSectionAxis == 1u)
	{
		vidxf.xyz = vidxf.yzx;
	}
	else if (crossSectionAxis == 2u)
	{
		vidxf.xyz = vidxf.zyx;
	}

	float3 ridxf = VirtualToReal(vidxf);

	float3 uvw = valueParams.dimInv.xyz * ridxf;

	float4 color = valueSRV.SampleLevel(borderSampler,uvw,0);
	if (pointFilter == 1u)
	{
		color = valueSRV.SampleLevel(borderPointSampler, uvw, 0);
	}

	color = colorMap_layer0(color);

	color.rgb *= color.a * intensityScale;

	color.rgb += (1.f - color.a) * backgroundColor.rgb;

	return color;
}