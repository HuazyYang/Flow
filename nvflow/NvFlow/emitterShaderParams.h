#ifndef HEREAFTER_EMITTERSHADERPARAMS_H
#define HEREAFTER_EMITTERSHADERPARAMS_H

/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//// global params
// struct GlobalParams
//{
//	NvFlowFloat4x4 gridToWorld;
//	NvFlowUint materialIdx;
//	NvFlowUint matPad0;
//	NvFlowUint matPad1;
//	NvFlowUint matPad2;
// };

//// AABB params
// struct AABBParams
//{
//	NvFlowUint3 boundMinMax;		// min in 16 lsbs, max in 16 msbs
//	NvFlowUint emitterID;			// emitter index for this frame
//	NvFlowUint emitParamsOffset;	// offset into stream where emitter parameters are located
//	NvFlowUint geomParamsOffset;	// offset into stream where geometry parameters are located, end of emitParams
//	NvFlowUint shapeParamsOffset;	// offset into stream where shape parameters are located, end of geomParams
//	NvFlowUint endParamsOffset;		// offset into stream where shape parameters end
// };

#define EMIT_AABB_boundMinMax 0u
#define EMIT_AABB_offsets 1u

#define EMIT_AABB_SIZE 2u

#if EMIT_DEFINES

#define emit_aabb_boundMinMax(ptr) ptr[EMIT_AABB_boundMinMax].xyz
#define emit_aabb_emitterID(ptr) ptr[EMIT_AABB_boundMinMax].w
#define emit_aabb_emitParamsOffset(ptr) ptr[EMIT_AABB_offset].x
#define emit_aabb_geomParamsOffset(ptr) ptr[EMIT_AABB_offset].y
#define emit_aabb_shapeParamsOffset(ptr) ptr[EMIT_AABB_offset].z
#define emit_aabb_endParamsOffset(ptr) ptr[EMIT_AABB_offset].w

#endif

//// Emitter params
// struct EmitParams
//{
//   NvFlowFloat4x4 gridToEmitter;
//	NvFlowFloat4x4 valueTransform;
//
//	NvFlowUint shapeRangeOffset;
//	NvFlowUint shapeRangeSize;
//   NvFlowUint materialIdx;
//   NvFlowUint pad1
//
//	float minActiveDist;
//	float maxActiveDist;
//	float minEdgeDist;
//	float maxEdgeDist;
//
//	NvFlowUint3 minVidx;
//	NvFlowUint numSubSteps_removed;
//	NvFlowUint3 maxVidx;
//	NvFlowUint shapeType;
//	NvFlowFloat3 emitterDimInv;
//   float shapeDistScale;
//
//	NvFlowFloat4 valueLinear;
//	NvFlowFloat4 valueAngular;
//	NvFlowFloat4 valueCenter;
//
//	NvFlowFloat4 valueCoupleRate;
//	NvFlowFloat4 deltaCoupleRate;
//	NvFlowFloat4 deltaTime;
//
//	float slipThickness;
//	float slipFactor;
//	float fuelReleaseTemp;
//	float fuelRelease;
// };

#define EMIT_DATA_gridToEmitter 0u
#define EMIT_DATA_valueTransform 4u
#define EMIT_DATA_shapeRange 8u
#define EMIT_DATA_minActiveDist 9u
#define EMIT_DATA_minVidx 10u
#define EMIT_DATA_maxVidx 11u
#define EMIT_DATA_emitterDimInv 12u
#define EMIT_DATA_valueLinear 13u
#define EMIT_DATA_valueAngular 14u
#define EMIT_DATA_valueCenter 15u
#define EMIT_DATA_valueCoupleRate 16u
#define EMIT_DATA_deltaTime 17u
#define EMIT_DATA_slip 18u

#define EMIT_DATA_SIZE 20u

#if EMIT_DEFINES

#define emit_data_gridToEmitter(ptr)                                                                                   \
  float4x4(ptr[EMIT_DATA_gridToEmitter + 0], ptr[EMIT_DATA_gridToEmitter + 1], ptr[EMIT_DATA_gridToEmitter + 2],       \
           ptr[EMIT_DATA_gridToEmitter + 3])

#define emit_data_valueTransform(ptr)                                                                                  \
  float4x4(ptr[EMIT_DATA_valueTransform + 0], ptr[EMIT_DATA_valueTransform + 1], ptr[EMIT_DATA_valueTransform + 2],    \
           ptr[EMIT_DATA_valueTransform + 3])

#define emit_data_shapeRangeOffset(ptr) asuint(ptr[EMIT_DATA_shapeRange].x)
#define emit_data_shapeRangeSize(ptr) asuint(ptr[EMIT_DATA_shapeRange].y)
#define emit_data_materialIdx(ptr) asuint(ptr[EMIT_DATA_shapeRange].z)

#define emit_data_minActiveDist(ptr) ptr[EMIT_DATA_minActiveDist].x
#define emit_data_maxActiveDist(ptr) ptr[EMIT_DATA_minActiveDist].y
#define emit_data_minEdgeDistInv(ptr) ptr[EMIT_DATA_minActiveDist].z
#define emit_data_maxEdgeDistInv(ptr) ptr[EMIT_DATA_minActiveDist].w

#define emit_data_minVidx(ptr) asuint(ptr[EMIT_DATA_minVidx].xyz)
#define emit_data_numSubSteps_removed(ptr) asuint(ptr[EMIT_DATA_minVidx].w)
#define emit_data_maxVidx(ptr) asuint(ptr[EMIT_DATA_maxVidx].xyz)
#define emit_data_shapeType(ptr) asuint(ptr[EMIT_DATA_maxVidx].w)
#define emit_data_emitterDimInv(ptr) ptr[EMIT_DATA_emitterDimInv].xyz
#define emit_data_shapeDistScale(ptr) ptr[EMIT_DATA_emitterDimInv].w

#define emit_data_valueLinear(ptr) ptr[EMIT_DATA_valueLinear]
#define emit_data_valueAngular(ptr) ptr[EMIT_DATA_valueAngular]
#define emit_data_valueCenter(ptr) ptr[EMIT_DATA_valueCenter]
#define emit_data_valueCoupleRate(ptr) ptr[EMIT_DATA_valueCoupleRate]
#define emit_data_deltaTime(ptr) ptr[EMIT_DATA_deltaTime]

#define emit_data_slipThickness(ptr) ptr[EMIT_DATA_slip].x
#define emit_data_slipFactor(ptr) ptr[EMIT_DATA_slip].y
#define emit_data_fuelReleaseTemp(ptr) ptr[EMIT_DATA_slip].z
#define emit_data_fuelRelease(ptr) ptr[EMIT_DATA_slip].w

#endif

#if EMIT_CPU_DEFINES

#define emit_data_gridToEmitter(ptr) *(NvFlowFloat4x4 *)(&ptr[EMIT_DATA_gridToEmitter])
#define emit_data_valueTransform(ptr) *(NvFlowFloat4x4 *)(&ptr[EMIT_DATA_valueTransform])

#define emit_data_shapeRangeOffset(ptr) *(NvFlowUint *)(&ptr[EMIT_DATA_shapeRange].x)
#define emit_data_shapeRangeSize(ptr) *(NvFlowUint *)(&ptr[EMIT_DATA_shapeRange].y)
#define emit_data_materialIdx(ptr) *(NvFlowUint *)(&ptr[EMIT_DATA_shapeRange].z)

#define emit_data_minActiveDist(ptr) ptr[EMIT_DATA_minActiveDist].x
#define emit_data_maxActiveDist(ptr) ptr[EMIT_DATA_minActiveDist].y
#define emit_data_minEdgeDistInv(ptr) ptr[EMIT_DATA_minActiveDist].z
#define emit_data_maxEdgeDistInv(ptr) ptr[EMIT_DATA_minActiveDist].w

#define emit_data_minVidx(ptr) *(NvFlowUint3 *)(&ptr[EMIT_DATA_minVidx].x)
#define emit_data_numSubSteps_removed(ptr) *(NvFlowUint *)(&ptr[EMIT_DATA_minVidx].w)
#define emit_data_maxVidx(ptr) *(NvFlowUint3 *)(&ptr[EMIT_DATA_maxVidx].x)
#define emit_data_shapeType(ptr) *(NvFlowUint *)(&ptr[EMIT_DATA_maxVidx].w)
#define emit_data_emitterDimInv(ptr) *(NvFlowFloat3 *)(&ptr[EMIT_DATA_emitterDimInv].x)
#define emit_data_shapeDistScale(ptr) ptr[EMIT_DATA_emitterDimInv].w

#define emit_data_valueLinear(ptr) ptr[EMIT_DATA_valueLinear]
#define emit_data_valueAngular(ptr) ptr[EMIT_DATA_valueAngular]
#define emit_data_valueCenter(ptr) ptr[EMIT_DATA_valueCenter]
#define emit_data_valueCoupleRate(ptr) ptr[EMIT_DATA_valueCoupleRate]
#define emit_data_deltaTime(ptr) ptr[EMIT_DATA_deltaTime]

#define emit_data_slipThickness(ptr) ptr[EMIT_DATA_slip].x
#define emit_data_slipFactor(ptr) ptr[EMIT_DATA_slip].y
#define emit_data_fuelReleaseTemp(ptr) ptr[EMIT_DATA_slip].z
#define emit_data_fuelRelease(ptr) ptr[EMIT_DATA_slip].w

#endif

//// shape params
// struct ShapeParams
//{
//	NvFlowFloat4 shapeData;
// };

#define EMIT_SHAPE_shapeData 0u

#define EMIT_SHAPE_SIZE 1u

#if EMIT_DEFINES

#define emit_shape_shapeData(ptr, offset) ptr[EMIT_SHAPE_shapeData + offset]

#endif

#if EMIT_CPU_DEFINES

#define emit_shape_shapeData(ptr) ptr[EMIT_SHAPE_shapeData]

#endif

// shared memory cache sizing
#if ENABLE_SIMPLE_EMIT
#define EMIT_SHAPE_CACHE_SIZE 16u
#else
#define EMIT_SHAPE_CACHE_SIZE 256u
#endif

#endif /* HEREAFTER_EMITTERSHADERPARAMS_H */
