/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef LAYER_NAME
#define LAYER_NAME layer0
#endif

#define _LAYER_NAME_FUNC(name, layerName) name ## layerName
#define LAYER_NAME_FUNC(name, layerName) _LAYER_NAME_FUNC(name, layerName)

#define flame_layerX			LAYER_NAME_FUNC(flame_, LAYER_NAME)
#define flameShadow_layerX		LAYER_NAME_FUNC(flameShadow_, LAYER_NAME)
#define rainbow_layerX			LAYER_NAME_FUNC(rainbow_, LAYER_NAME)
#define debugColor_layerX		LAYER_NAME_FUNC(debugColor_, LAYER_NAME)

#define colorMapSRV_layerX		LAYER_NAME_FUNC(colorMapSRV_, LAYER_NAME)
#define colorMap_layerX			LAYER_NAME_FUNC(colorMap_, LAYER_NAME)
#define densitySRV_layerX		LAYER_NAME_FUNC(densitySRV_, LAYER_NAME)
#define rayMarchSample_layerX	LAYER_NAME_FUNC(rayMarchSample_, LAYER_NAME)

#define colorMapRange_layerX	LAYER_NAME_FUNC(colorMapRange_, LAYER_NAME)
#define alphaBias_layerX		LAYER_NAME_FUNC(alphaBias_, LAYER_NAME)
#define intensityBias_layerX	LAYER_NAME_FUNC(intensityBias_, LAYER_NAME)
#define additiveFactor_layerX	LAYER_NAME_FUNC(additiveFactor_, LAYER_NAME)

#define colorMapCompMask_layerX		LAYER_NAME_FUNC(colorMapCompMask_, LAYER_NAME)
#define alphaCompMask_layerX		LAYER_NAME_FUNC(alphaCompMask_, LAYER_NAME)
#define intensityCompMask_layerX	LAYER_NAME_FUNC(intensityCompMask_, LAYER_NAME)

float4 flame_layerX(float4 value)
{
	float colorMapX = dot(value, colorMapCompMask_layerX);
	float alphaMod = dot(value, alphaCompMask_layerX) + alphaBias_layerX;
	float intensityMod = dot(value, intensityCompMask_layerX) + intensityBias_layerX;
	// scale sample location based on colorMap range
	float u = colorMapRange_layerX.y * (colorMapX - colorMapRange_layerX.x);
	float4 color = colorMapSRV_layerX.SampleLevel(clampSampler, u, 0);
	color.rgb *= intensityMod;
	return float4(color.rgb, color.a * alphaMod);
}

float4 rainbow_layerX(float4 value)
{
	float4 color;
	color.r = abs(24.f * (frac(value.a + 0.00f) - 0.5f)) - 6.f;
	color.g = abs(24.f * (frac(value.a + 0.33f) - 0.5f)) - 6.f;
	color.b = abs(24.f * (frac(value.a + 0.66f) - 0.5f)) - 6.f;
	color.a = value.a;
	return color;
}

float4 debugColor_layerX(float4 value)
{
	float len = length(value.rgb);
	float4 color;
	if (len > 0.f)
	{
		color.rgba = float4(abs(value.rgb) / len, 2.f*len);
	}
	else
	{
		color.rgba = 0.f.xxxx;
	}
	return color;
}

float4 colorMap_layerX(float4 value)
{
	float4 color = value;
#if RAW_ONLY

#elif COLORMAP_ONLY
	color = flame_layerX(value);
#elif DEBUGCOLOR_ONLY
	if (renderMode.x == 2) color = rainbow_layerX(value);
	if (renderMode.x == 3) color = debugColor_layerX(value);
#else		
	if (renderMode.x == 0) { color = flame_layerX(value); }
	if (renderMode.x == 1) { color = value; }
	if (renderMode.x == 2) { color = rainbow_layerX(value); }
	if (renderMode.x == 3) { color = debugColor_layerX(value); }
#endif

	return color;
}

#if RAY_MARCH_SAMPLE

void rayMarchSample_layerX(inout float4 sum, float3 uvw, float tmin, int i, float alphaScaleIn)
{
	float4 color = densitySRV_layerX.SampleLevel(borderSampler, uvw, 0);

	color = colorMap_layerX(color);

	// clamp color to valid range
	color.rgb = max(0.f.xxx, color.rgb);
	color.a = saturate(color.a);

	float alphaScaleLocal = alphaScaleIn;
	// fade with low t values
	float fadet = max(0.f, tmin + float(i) + eyeFadeOffset);
	if (fadet < eyeFade)
	{
		alphaScaleLocal *= eyeFadeInv * fadet;
	}

	// scale alpha
	color.a *= alphaScaleLocal;
	// blend
	sum.rgb = sum.a * (color.a * color.rgb) + sum.rgb;
	sum.a = (1.f - color.a * saturate(1.f - additiveFactor_layerX)) * sum.a;
}

#endif

#undef flame_layerX
#undef flameShadow_layerX
#undef rainbow_layerX
#undef debugColor_layerX

#undef colorMapSRV_layerX
#undef colorMap_layerX
#undef densitySRV_layerX
#undef rayMarchSample_layerX

#undef colorMapRange_layerX
#undef alphaBias_layerX
#undef intensityBias_layerX
#undef additiveFactor_layerX

#undef colorMapCompMask_layerX
#undef alphaCompMask_layerX
#undef intensityCompMask_layerX

#undef LAYER_NAME_FUNC