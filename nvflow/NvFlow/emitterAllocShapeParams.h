/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#define EMIT_ALLOC_SHAPE_BATCH_SIZE 32u
#define EMIT_ALLOC_SHAPE_CACHE_SIZE 512u
#define EMIT_ALLOC_SHAPE_SDF_CACHE_SIZE 1u

/*
struct EmitAllocPerShape
{
	NvFlowFloat4x4 gridToEmitter;

	NvFlowUint3 minAllocMaskIdx;
	NvFlowUint shapeType;
	NvFlowUint3 maxAllocMaskIdx;
	float shapeDistScale;

	NvFlowUint shapeRangeOffset;
	NvFlowUint shapeRangeSize;
	float minActiveDist;
	float maxActiveDist;
};
*/

#define EMIT_ALLOC_SHAPE_gridToEmitter 0u

#define EMIT_ALLOC_SHAPE_minAllocMaskIdx 4u
#define EMIT_ALLOC_SHAPE_maxAllocMaskIdx 5u

#define EMIT_ALLOC_SHAPE_shapeRange 6u
#define EMIT_ALLOC_SHAPE_materialIdx 7u

#define EMIT_ALLOC_SHAPE_SIZE 8u

#if EMIT_ALLOC_SHAPE_DEFINES

#define emit_alloc_shape_gridToEmitter(ptr, off) float4x4( \
	ptr[EMIT_ALLOC_SHAPE_gridToEmitter+0+off],\
	ptr[EMIT_ALLOC_SHAPE_gridToEmitter+1+off],\
	ptr[EMIT_ALLOC_SHAPE_gridToEmitter+2+off],\
	ptr[EMIT_ALLOC_SHAPE_gridToEmitter+3+off])

#define emit_alloc_shape_minAllocMaskIdx(ptr, off) asuint(ptr[EMIT_ALLOC_SHAPE_minAllocMaskIdx+off].xyz)
#define emit_alloc_shape_shapeType(ptr, off) asuint(ptr[EMIT_ALLOC_SHAPE_minAllocMaskIdx+off].w)
#define emit_alloc_shape_maxAllocMaskIdx(ptr, off) asuint(ptr[EMIT_ALLOC_SHAPE_maxAllocMaskIdx+off].xyz)
#define emit_alloc_shape_shapeDistScale(ptr, off) ptr[EMIT_ALLOC_SHAPE_maxAllocMaskIdx+off].w

#define emit_alloc_shape_shapeRangeOffset(ptr, off) asuint(ptr[EMIT_ALLOC_SHAPE_shapeRange+off].x)
#define emit_alloc_shape_shapeRangeSize(ptr, off)   asuint(ptr[EMIT_ALLOC_SHAPE_shapeRange+off].y)
#define emit_alloc_shape_minActiveDist(ptr, off)   ptr[EMIT_ALLOC_SHAPE_shapeRange+off].z
#define emit_alloc_shape_maxActiveDist(ptr, off)   ptr[EMIT_ALLOC_SHAPE_shapeRange+off].w

#define emit_alloc_shape_materialIdx(ptr, off) asuint(ptr[EMIT_ALLOC_SHAPE_materialIdx+off].x)

#endif

#if EMIT_ALLOC_SHAPE_CPU_DEFINES

#define emit_alloc_shape_gridToEmitter(ptr)		*(NvFlowFloat4x4*)(&ptr[EMIT_ALLOC_SHAPE_gridToEmitter])

#define emit_alloc_shape_minAllocMaskIdx(ptr)	*(NvFlowUint3*)(&ptr[EMIT_ALLOC_SHAPE_minAllocMaskIdx].x)
#define emit_alloc_shape_shapeType(ptr)			*(NvFlowUint*)(&ptr[EMIT_ALLOC_SHAPE_minAllocMaskIdx].w)
#define emit_alloc_shape_maxAllocMaskIdx(ptr)	*(NvFlowUint3*)(&ptr[EMIT_ALLOC_SHAPE_maxAllocMaskIdx].x)
#define emit_alloc_shape_shapeDistScale(ptr)	ptr[EMIT_ALLOC_SHAPE_maxAllocMaskIdx].w

#define emit_alloc_shape_shapeRangeOffset(ptr)	*(NvFlowUint*)(&ptr[EMIT_ALLOC_SHAPE_shapeRange].x)
#define emit_alloc_shape_shapeRangeSize(ptr)	*(NvFlowUint*)(&ptr[EMIT_ALLOC_SHAPE_shapeRange].y)
#define emit_alloc_shape_minActiveDist(ptr)		ptr[EMIT_ALLOC_SHAPE_shapeRange].z
#define emit_alloc_shape_maxActiveDist(ptr)		ptr[EMIT_ALLOC_SHAPE_shapeRange].w

#define emit_alloc_shape_materialIdx(ptr) *(NvFlowUint*)(&ptr[EMIT_ALLOC_SHAPE_materialIdx].x)

#endif

/*
struct EmitAllocShapeParams
{
	NvFlowFloat4 gridDimInv;
	NvFlowUint4 numShapes;

	EmitAllocPerShape shapes[EMIT_ALLOC_SHAPE_BATCH_SIZE];

	NvFlowFloat4 shapeData[EMIT_ALLOC_SHAPE_CACHE_SIZE];
};
*/

struct EmitAllocShapeShaderParams
{
	NvFlowFloat4 gridDimInv;
	NvFlowUint4 numShapes;
	NvFlowUint4 blockIdxOffset;
	NvFlowUint materialIdx;
	NvFlowUint matPad0;
	NvFlowUint matPad1;
	NvFlowUint matPad2;
};