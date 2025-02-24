/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//NvFlowFloat4x4 gridToEmitter1;
//NvFlowFloat4x4 gridToEmitter2;

//NvFlowUint4 minVblockIdx;
//NvFlowUint4 maxVblockIdx;

//NvFlowFloat4 vGridDimInv;

#define EMITALLOC_OFF_gridToEmitter1 0u
#define EMITALLOC_OFF_gridToEmitter2 4u

#define EMITALLOC_OFF_minVblockIdx 8u
#define EMITALLOC_OFF_maxVblockIdx 9u

#define EMITALLOC_OFF_vGridDimInv 10u
#define EMITALLOC_OFF_materialIdx 11u

#define EMITALLOC_DATA_SIZE 12u

#if EMITALLOC_DEFINES

#define gridToEmitter1 float4x4( \
	sdata[EMITALLOC_OFF_gridToEmitter1+0],\
	sdata[EMITALLOC_OFF_gridToEmitter1+1],\
	sdata[EMITALLOC_OFF_gridToEmitter1+2],\
	sdata[EMITALLOC_OFF_gridToEmitter1+3])

#define gridToEmitter2 float4x4( \
	sdata[EMITALLOC_OFF_gridToEmitter2+0],\
	sdata[EMITALLOC_OFF_gridToEmitter2+1],\
	sdata[EMITALLOC_OFF_gridToEmitter2+2],\
	sdata[EMITALLOC_OFF_gridToEmitter2+3])

#define minVblockIdx asuint(sdata[EMITALLOC_OFF_minVblockIdx])
#define maxVblockIdx asuint(sdata[EMITALLOC_OFF_maxVblockIdx])

#define vGridDimInv sdata[EMITALLOC_OFF_vGridDimInv]
#define emitalloc_materialIdx asuint(sdata[EMITALLOC_OFF_materialIdx].x)

#endif