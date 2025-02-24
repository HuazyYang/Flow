/*
* Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef NUM_LAYERS
#define NUM_LAYERS 1
#endif

#ifndef THREAD_DIM_X

#define THREAD_DIM_X 8
#define THREAD_DIM_Y 8
#define THREAD_DIM_Z 8

#define SIMULATE_ONCE 1

#endif

#include "frameworkHybrid.hlsli"

/// End NvFlow address translation utilities

cbuffer params : register(b0)
{
	NvFlowShaderLinearParams exportParams;
	NvFlowShaderLinearParams importParams;
	NvFlowUint4 renderMode;
	float alphaBias_layer0;
	float intensityBias_layer0;
	float pad1;
	float pad2;
	NvFlowFloat4 colorMapCompMask_layer0;
	NvFlowFloat4 colorMapRange_layer0;
	NvFlowFloat4 alphaCompMask_layer0;
	NvFlowFloat4 intensityCompMask_layer0;
};

Buffer<uint> exportBlockList : register(t0);
Texture3D<uint> exportBlockTable : register(t1);
Texture3D<float4> exportData : register(t2);

Buffer<uint> importBlockList : register(t3);
Texture3D<uint> importBlockTable : register(t4);
RWTexture3D<float4> importDataRW : register(u0);

Texture1D<float4> colorMapSRV_layer0 : register(t5);

//DISPATCH_ID_TO_VIRTUAL(importBlockList, importParams);

DISPATCH_ID_TO_VIRTUAL_AND_REAL(importBlockList, importBlockTable, importParams);

VIRTUAL_TO_REAL_LINEAR(VirtualToRealExport, exportBlockTable, exportParams);
//VIRTUAL_TO_REAL_IMPORT(VirtualToRealImport, importBlockTable, importParams);

#define LAYER_NAME layer0
#include "volumeRenderColormap.hlsli"
#undef LAYER_NAME

[numthreads(THREAD_DIM_X, THREAD_DIM_Y, THREAD_DIM_Z)]
void volumeRenderLightingCS(uint3 tidx : SV_DispatchThreadID)
{
	int3 vidx;
	int3 ridx;
	DispatchIDToVirtualAndReal(tidx, vidx, ridx);

	float3 vidxf = float3(vidx)+0.5f.xxx;

	float3 vidxNorm = 2.f * vidxf * exportParams.vdimInv.xyz - 1.f;

	float3 ridxExport = VirtualToRealExport(vidxf);
	float4 value = exportData.SampleLevel(borderSampler, exportParams.dimInv.xyz * ridxExport, 0);
	
	float4 color = colorMap_layer0(value);

	//int3 ridxImport = VirtualToRealImport(vidx);
	importDataRW[ridx] = color;
}