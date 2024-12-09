/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

float opacity_func(float dist)
{
	float opacity = 0.f;
	if (dist >= emit_data_minActiveDist(sdata) && dist <= emit_data_maxActiveDist(sdata))
	{
		float mindx = (dist - emit_data_minActiveDist(sdata));
		float maxdx = (emit_data_maxActiveDist(sdata) - dist);
		opacity = min(
		(mindx != 0.f) ? saturate(mindx * emit_data_minEdgeDistInv(sdata)) : 0.f,
		(maxdx != 0.f) ? saturate(maxdx * emit_data_maxEdgeDistInv(sdata)) : 0.f 
		);
	}
	return opacity;
}

float sdfDist(float3 emitterLocal)
{
	float3 uvw = 0.5f * emitterLocal + 0.5f;
	return sdf_SRV.SampleLevel(clampSampler, uvw, 0);
}

float sphereDist(float3 emitterLocal)
{
	float d = length(emitterLocal);
	float v = d - emit_shape_shapeData(sshape, 0u).x;
	return v;
}

float boxDist(float3 emitterLocal)
{
	float3 dr = abs(emitterLocal) - emit_shape_shapeData(sshape, 0u).xyz;
	float v = max(dr.x, max(dr.y, dr.z));
	return v;
}

float capsuleDist(float3 emitterLocal)
{
	emitterLocal.x = max(0.f, abs(emitterLocal.x) - 0.5f * emit_shape_shapeData(sshape, 0u).y);
	float d = length(emitterLocal);
	float v = d - emit_shape_shapeData(sshape, 0u).x;
	return v;
}

float convexDist(float3 emitterLocal)
{
	NvFlowUint shapeOffset = EMIT_SHAPE_SIZE * emit_data_shapeRangeOffset(sdata);
	NvFlowUint shapeCount = min(EMIT_SHAPE_CACHE_SIZE, emit_data_shapeRangeSize(sdata));
	float dist = -1.f;
	if (0u < shapeCount)
	{
		float4 plane = emit_shape_shapeData(sshape, 0u * EMIT_SHAPE_SIZE);
		dist = dot(emitterLocal, plane.xyz) - plane.w;
	}
	for (NvFlowUint i = 1u; i < shapeCount; i++)
	{
		float4 plane = emit_shape_shapeData(sshape, i * EMIT_SHAPE_SIZE);
		float v = dot(emitterLocal, plane.xyz) - plane.w;
		dist = max(dist, v);
	}
	return dist;
}

float sampleDist(float3 emitterLocal)
{
	float dist = 1.f;
	uint type = emit_data_shapeType(sdata);

	// only support SDFs outside of MacCormack
#if !ENABLE_EMIT
	if (type == 0) dist = sdfDist(emitterLocal);
#endif

#if ENABLE_SIMPLE_EMIT
	if (type == 1) dist = sphereDist(emitterLocal);
	if (type == 2) dist = boxDist(emitterLocal);
	if (type == 3) dist = capsuleDist(emitterLocal);
#else
	if (type == 1) dist = sphereDist(emitterLocal);
	if (type == 2) dist = boxDist(emitterLocal);
	if (type == 3) dist = capsuleDist(emitterLocal);
	if (type == 4) dist = convexDist(emitterLocal);
#endif

	dist *= emit_data_shapeDistScale(sdata);

	return dist;
}

float3 sampleGradient(float3 emitterLocal)
{
	float3 dg;
	dg.x = 0.5f * (
		sampleDist(emitterLocal + float3(emit_data_emitterDimInv(sdata).x, 0.f, 0.f)) - 
		sampleDist(emitterLocal - float3(emit_data_emitterDimInv(sdata).x, 0.f, 0.f))
		);
	dg.y = 0.5f * (
		sampleDist(emitterLocal + float3(0.f, emit_data_emitterDimInv(sdata).y, 0.f)) - 
		sampleDist(emitterLocal - float3(0.f, emit_data_emitterDimInv(sdata).y, 0.f))
		);
	dg.z = 0.5f * (
		sampleDist(emitterLocal + float3(0.f, 0.f, emit_data_emitterDimInv(sdata).z)) - 
		sampleDist(emitterLocal - float3(0.f, 0.f, emit_data_emitterDimInv(sdata).z))
		);
	return dg;
}

float4 emitDensityFunction(float4 value, float coarseTemp, float4 grid_ndc)
{
	float4 emitterLocal = mul(grid_ndc, emit_data_gridToEmitter(sdata));
	float dist = sampleDist(emitterLocal.xyz);

	float opacity = opacity_func(dist);

	float4 valueRate = saturate(emit_data_deltaTime(sdata) * emit_data_valueCoupleRate(sdata) * opacity);

	float4 emitValue = emit_data_valueLinear(sdata);

	if ((emit_data_materialIdx(sdata) == materialIdx) ||
		(emit_data_materialIdx(sdata) == 0u))
	{
		float4 emitValueTemp = emitValue;

		if (emit_data_fuelRelease(sdata) != 0.f)
		{
			float burnTemp = max(coarseTemp, value.x);
			if (burnTemp > emit_data_fuelReleaseTemp(sdata))
			{
				emitValueTemp.y += emit_data_fuelRelease(sdata);
			}
		}

		value += valueRate * (emitValueTemp - value);
	}

	return value;
}

float4 emitVelocityFunction(float4 value, float4 grid_ndc)
{
	float3 grad = 0.f.xxx;
	float slip_t = 0.f;

	bool checkSlip = (emit_data_slipFactor(sdata) > 0.f && emit_data_slipThickness(sdata) > 0.f);

	float4 emitterLocal = mul(grid_ndc, emit_data_gridToEmitter(sdata));
	float dist = sampleDist(emitterLocal.xyz);

	bool shouldSlip = checkSlip &&
		dist >= emit_data_maxActiveDist(sdata) &&
		dist <= emit_data_maxActiveDist(sdata) + emit_data_slipThickness(sdata);
	if (shouldSlip)
	{
		slip_t = 1.f - (dist - emit_data_maxActiveDist(sdata)) / emit_data_slipThickness(sdata);
		grad = sampleGradient(emitterLocal.xyz);
		grad = mul(float4(grad, 0.f), emit_data_valueTransform(sdata)).xyz;

		shouldSlip = (dot(grad, grad) > 0.f);
		if (shouldSlip)
		{
			grad = normalize(grad);
		}
	}

	float opacity = opacity_func(dist);

	float4 valueRate = saturate(emit_data_deltaTime(sdata) * emit_data_valueCoupleRate(sdata) * opacity);

	float4 world_coord = mul(grid_ndc, gridToWorld);
	float4 offset = world_coord - emit_data_valueCenter(sdata);

	float4 emitValue = emit_data_valueLinear(sdata);
	emitValue.x += +offset.z * emit_data_valueAngular(sdata).y - offset.y * emit_data_valueAngular(sdata).z;
	emitValue.y += -offset.z * emit_data_valueAngular(sdata).x + offset.x * emit_data_valueAngular(sdata).z;
	emitValue.z += +offset.y * emit_data_valueAngular(sdata).x - offset.x * emit_data_valueAngular(sdata).y;

	if ((emit_data_materialIdx(sdata) == materialIdx) ||
		(emit_data_materialIdx(sdata) == 0u))
	{
		value += valueRate * (emitValue - value);

		// apply slip constraint
		if (shouldSlip)
		{
			float d1 = dot(value.xyz, grad);
			float d2 = dot(emitValue.xyz, grad);

			value.xyz -= slip_t * emit_data_slipFactor(sdata) * (d1 - d2) * grad;
		}
	}

	return value;
}