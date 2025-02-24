// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2014-2021 NVIDIA Corporation. All rights reserved.

#include "scene.h"

#include "imgui.h"
#include "imguiser.h"

#include "traceUtils.h"

// ******************** FlowContext ************************

void FlowContext::init(AppGraphCtx* appctx)
{
	m_appctx = appctx;

	m_renderContext = TRACE(NvFlowInteropCreateContext(appctx));
	m_dsv = TRACE(NvFlowInteropCreateDepthStencilView(appctx, m_renderContext));
	m_rtv = TRACE(NvFlowInteropCreateRenderTargetView(appctx, m_renderContext));

	// establishes m_gridContext
	createComputeContext();
}

void FlowContext::createComputeContext()
{
	m_multiGPUSupported = TRACE(NvFlowDedicatedDeviceAvailable(m_renderContext));
	m_multiGPUActive = m_multiGPUSupported && m_enableMultiGPU;
	if (m_multiGPUActive)
	{
		NvFlowDeviceDesc deviceDesc = {};
		NvFlowDeviceDescDefaults(&deviceDesc);

		deviceDesc.mode = eNvFlowDeviceModeProxy;
		m_renderDevice = TRACE(NvFlowCreateDevice(m_renderContext, &deviceDesc));
		deviceDesc.mode = eNvFlowDeviceModeUnique;
		m_gridDevice = TRACE(NvFlowCreateDevice(m_renderContext, &deviceDesc));

		NvFlowDeviceQueueDesc deviceQueueDesc = {};
		deviceQueueDesc.queueType = eNvFlowDeviceQueueTypeGraphics;
		deviceQueueDesc.lowLatency = false;
		m_gridQueue = TRACE(NvFlowCreateDeviceQueue(m_gridDevice, &deviceQueueDesc));
		deviceQueueDesc.queueType = eNvFlowDeviceQueueTypeCopy;
		m_gridCopyQueue = TRACE(NvFlowCreateDeviceQueue(m_gridDevice, &deviceQueueDesc));
		m_renderCopyQueue = TRACE(NvFlowCreateDeviceQueue(m_renderDevice, &deviceQueueDesc));

		m_gridContext = TRACE(NvFlowDeviceQueueCreateContext(m_gridQueue));
		m_gridCopyContext = TRACE(NvFlowDeviceQueueCreateContext(m_gridCopyQueue));
		m_renderCopyContext = TRACE(NvFlowDeviceQueueCreateContext(m_renderCopyQueue));

		NvFlowDeviceQueueStatus status = {};
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
	}
	else
	{
		m_commandQueueSupported = TRACE(NvFlowDedicatedDeviceQueueAvailable(m_renderContext));
		m_commandQueueActive = m_commandQueueSupported && m_enableCommandQueue;
		if (m_commandQueueActive)
		{
			NvFlowDeviceDesc deviceDesc = {};
			NvFlowDeviceDescDefaults(&deviceDesc);

			deviceDesc.mode = eNvFlowDeviceModeProxy;
			m_renderDevice = TRACE(NvFlowCreateDevice(m_renderContext, &deviceDesc));
			m_gridDevice = m_renderDevice;

			NvFlowDeviceQueueDesc deviceQueueDesc = {};
			deviceQueueDesc.queueType = eNvFlowDeviceQueueTypeCompute;
			deviceQueueDesc.lowLatency = true;
			m_gridQueue = TRACE(NvFlowCreateDeviceQueue(m_gridDevice, &deviceQueueDesc));
			deviceQueueDesc.queueType = eNvFlowDeviceQueueTypeCopy;
			m_gridCopyQueue = TRACE(NvFlowCreateDeviceQueue(m_gridDevice, &deviceQueueDesc));
			m_renderCopyQueue = m_gridCopyQueue;

			m_gridContext = TRACE(NvFlowDeviceQueueCreateContext(m_gridQueue));
			m_gridCopyContext = TRACE(NvFlowDeviceQueueCreateContext(m_gridCopyQueue));
			m_renderCopyContext = m_gridCopyContext;

			NvFlowDeviceQueueStatus status = {};
			TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
		}
		else
		{
			m_gridContext = m_renderContext;
			m_gridCopyContext = m_renderContext;
			m_renderCopyContext = m_renderContext;
		}
	}
}

void FlowContext::release()
{
	TRACE(NvFlowReleaseRenderTargetView(m_rtv));
	TRACE(NvFlowReleaseDepthStencilView(m_dsv));
	TRACE(NvFlowReleaseContext(m_renderContext));

	releaseComputeContext();
}

void FlowContext::releaseComputeContext()
{
	if (m_gridDevice != m_renderDevice)
	{
		TRACE(NvFlowReleaseContext(m_gridContext));
		TRACE(NvFlowReleaseContext(m_gridCopyContext));
		TRACE(NvFlowReleaseContext(m_renderCopyContext));
		m_gridContext = nullptr;
		m_gridCopyContext = nullptr;
		m_renderCopyContext = nullptr;

		TRACE(NvFlowReleaseDeviceQueue(m_gridQueue));
		TRACE(NvFlowReleaseDeviceQueue(m_gridCopyQueue));
		TRACE(NvFlowReleaseDeviceQueue(m_renderCopyQueue));
		m_gridQueue = nullptr;
		m_gridCopyQueue = nullptr;
		m_renderCopyQueue = nullptr;

		TRACE(NvFlowReleaseDevice(m_gridDevice));
		TRACE(NvFlowReleaseDevice(m_renderDevice));
		m_gridDevice = nullptr;
		m_renderDevice = nullptr;
	}
	else if (m_gridContext != m_renderContext)
	{
		TRACE(NvFlowReleaseContext(m_gridContext));
		TRACE(NvFlowReleaseContext(m_gridCopyContext));
		m_gridContext = nullptr;
		m_gridCopyContext = nullptr;
		m_renderCopyContext = nullptr;

		TRACE(NvFlowReleaseDeviceQueue(m_gridQueue));
		TRACE(NvFlowReleaseDeviceQueue(m_gridCopyQueue));
		m_gridQueue = nullptr;
		m_gridCopyQueue = nullptr;
		m_renderCopyQueue = nullptr;

		TRACE(NvFlowReleaseDevice(m_gridDevice));
		m_gridDevice = nullptr;
		m_renderDevice = nullptr;
	}
	else
	{
		m_gridContext = nullptr;
		m_gridCopyContext = nullptr;
		m_renderCopyContext = nullptr;

		m_gridQueue = nullptr;
		m_gridCopyQueue = nullptr;
		m_renderCopyQueue = nullptr;

		m_gridDevice = nullptr;
		m_renderDevice = nullptr;
	}
}

int FlowContext::computeContextBegin()
{
	int framesInFlight = 0u;
	if (m_gridDevice != m_renderDevice)
	{
		NvFlowDeviceQueueStatus status = {};
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
		framesInFlight = status.framesInFlight;

		TRACE(NvFlowDeviceQueueUpdateContext(m_gridCopyQueue, m_gridCopyContext, &status));
		TRACE(NvFlowDeviceQueueUpdateContext(m_renderCopyQueue, m_renderCopyContext, &status));
	}
	else if (m_gridContext != m_renderContext)
	{
		NvFlowDeviceQueueStatus status = {};
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
		framesInFlight = status.framesInFlight;

		TRACE(NvFlowDeviceQueueUpdateContext(m_gridCopyQueue, m_gridCopyContext, &status));
	}
	else
	{
		TRACE(NvFlowInteropUpdateContext(m_renderContext, m_appctx));
		TRACE(NvFlowContextPush(m_gridContext));
	}
	return framesInFlight;
}

void FlowContext::computeContextEnd()
{
	if (m_gridDevice != m_renderDevice)
	{
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridQueue, m_gridContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridCopyQueue, m_gridCopyContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_renderCopyQueue, m_renderCopyContext));
	}
	else if (m_gridContext != m_renderContext)
	{
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridQueue, m_gridContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridCopyQueue, m_gridCopyContext));
	}
	else
	{
		TRACE(NvFlowContextPop(m_gridContext));
	}
}

bool FlowContext::updateBegin(float dt)
{
	m_framesInFlight = computeContextBegin();
	bool shouldFlush = (m_framesInFlight < m_maxFramesInFlight);

	if (shouldFlush)
	{
		TRACE(NvFlowContextFlushRequestPush(m_gridContext));
		TRACE(NvFlowContextFlushRequestPush(m_gridCopyContext));
		TRACE(NvFlowContextFlushRequestPush(m_renderCopyContext));
	}

	m_statUpdateAttemptCount += 1.0;
	if (shouldFlush) m_statUpdateSuccessCount += 1.0;
	m_statUpdateAttemptCount *= 0.99;
	m_statUpdateSuccessCount *= 0.99;
	m_statUpdateDt = dt;

	return shouldFlush;
}

void FlowContext::updateEnd()
{
	computeContextEnd();
}

void FlowContext::preDrawBegin()
{
	if (m_gridDevice != m_renderDevice)
	{
		// update fence status on grid queue, no need to flush
		NvFlowDeviceQueueStatus status = {};
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridCopyQueue, m_gridCopyContext, &status));
		TRACE(NvFlowDeviceQueueUpdateContext(m_renderCopyQueue, m_renderCopyContext, &status));
	}
	else if (m_gridContext != m_renderContext)
	{
		// update fence status on grid queue, no need to flush
		NvFlowDeviceQueueStatus status = {};
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridQueue, m_gridContext, &status));
		TRACE(NvFlowDeviceQueueUpdateContext(m_gridCopyQueue, m_gridCopyContext, &status));
	}

	TRACE(NvFlowInteropUpdateContext(m_renderContext, m_appctx));
	TRACE(NvFlowContextPush(m_renderContext));
}

void FlowContext::preDrawEnd()
{
	TRACE(NvFlowContextPop(m_renderContext));

	// This will make gridProxy flush work
	if (m_gridDevice != m_renderDevice)
	{
		//TRACE(NvFlowDeviceQueueFlush(m_gridQueue, m_gridContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridCopyQueue, m_gridCopyContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_renderCopyQueue, m_renderCopyContext));
	}
	else if (m_gridContext != m_renderContext)
	{
		//TRACE(NvFlowDeviceQueueFlush(m_gridQueue, m_gridContext));
		TRACE(NvFlowDeviceQueueConditionalFlush(m_gridCopyQueue, m_gridCopyContext));
	}
}

void FlowContext::drawBegin()
{
	TRACE(NvFlowInteropUpdateContext(m_renderContext, m_appctx));
	TRACE(NvFlowInteropUpdateDepthStencilView(m_dsv, m_appctx, m_renderContext));
	TRACE(NvFlowInteropUpdateRenderTargetView(m_rtv, m_appctx, m_renderContext));
	TRACE(NvFlowContextPush(m_renderContext));
}

void FlowContext::drawEnd()
{
	TRACE(NvFlowContextPop(m_renderContext));
}

// ******************* FlowGridActor *************************

void FlowGridActor::initParams(size_t vramAmount)
{
	NvFlowGridDescDefaults(&m_gridDesc);
	NvFlowGridParamsDefaults(&m_gridParams);
	NvFlowGridMaterialParamsDefaults(&m_materialParams);
	NvFlowRenderMaterialParamsDefaults(&m_renderMaterialDefaultParams);
	NvFlowRenderMaterialParamsDefaults(&m_renderMaterialMat0Params);
	NvFlowRenderMaterialParamsDefaults(&m_renderMaterialMat1Params);
	NvFlowVolumeRenderParamsDefaults(&m_renderParams);
	NvFlowCrossSectionParamsDefaults(&m_crossSectionParams);

	m_renderMaterialMat0Params.material = NvFlowGridMaterialHandle{ nullptr, 1u };
	m_renderMaterialMat1Params.material = NvFlowGridMaterialHandle{ nullptr, 1u };

	// default VTR off on debug build
	#ifdef _DEBUG
	m_gridDesc.enableVTR = false;
	#endif

	// attempt to pick good memory limits based on vramAmount
	{
		if (vramAmount <= 2ull * 1024ull * 1024ull * 1024ull)
		{
			m_memoryLimit = 1.f;
		}
		else if (vramAmount <= 3ull * 1024ull * 1024ull * 1024ull)
		{
			m_memoryLimit = 2.f;
		}
	}

	m_gridDesc.residentScale = m_memoryLimit * m_memoryScale;

	// configure gravity
	m_gridParams.gravity = NvFlowFloat3{ 0.f, -1.f, 0.f };
}

void FlowGridActor::init(FlowContext* flowContext, AppGraphCtx* appctx)
{
	m_appctx = appctx;

	// create compute resources
	m_grid = TRACE(NvFlowCreateGrid(flowContext->m_gridContext, &m_gridDesc));

	auto proxyGridExport = TRACE(NvFlowGridGetGridExport(flowContext->m_gridContext, m_grid));

	NvFlowGridProxyDesc proxyDesc = {};
	proxyDesc.gridContext = flowContext->m_gridContext;
	proxyDesc.renderContext = flowContext->m_renderContext;
	proxyDesc.gridCopyContext = flowContext->m_gridCopyContext;
	proxyDesc.renderCopyContext = flowContext->m_renderCopyContext;
	proxyDesc.gridExport = proxyGridExport;
	proxyDesc.proxyType = eNvFlowGridProxyTypePassThrough;
	if (flowContext->m_multiGPUActive)
	{
		proxyDesc.proxyType = eNvFlowGridProxyTypeMultiGPU;
	}
	else if (flowContext->m_commandQueueActive)
	{
		proxyDesc.proxyType = eNvFlowGridProxyTypeInterQueue;
	}

	m_gridProxy = TRACE(NvFlowCreateGridProxy(&proxyDesc));

	auto gridExport = TRACE(NvFlowGridProxyGetGridExport(m_gridProxy, flowContext->m_renderContext));

	// create render resources
	NvFlowVolumeRenderDesc volumeRenderDesc;
	volumeRenderDesc.gridExport = gridExport;

	m_volumeRender = TRACE(NvFlowCreateVolumeRender(flowContext->m_renderContext, &volumeRenderDesc));

	NvFlowCrossSectionDesc crossSectionDesc = {};
	crossSectionDesc.gridExport = gridExport;
	
	m_crossSection = TRACE(NvFlowCreateCrossSection(flowContext->m_renderContext, &crossSectionDesc));

	NvFlowGridSummaryDesc gridSummaryDesc = {};
	gridSummaryDesc.gridExport = gridExport;

	m_gridSummary = TRACE(NvFlowCreateGridSummary(flowContext->m_gridContext, &gridSummaryDesc));
	m_gridSummaryStateCPU = TRACE(NvFlowCreateGridSummaryStateCPU(m_gridSummary));

	NvFlowRenderMaterialPoolDesc materialPoolDesc = {};
	materialPoolDesc.colorMapResolution = 64u;
	m_colorMap.m_materialPool = TRACE(NvFlowCreateRenderMaterialPool(flowContext->m_renderContext, &materialPoolDesc));

	NvFlowRenderMaterialParams materialParams = {};
	NvFlowRenderMaterialParamsDefaults(&materialParams);
	m_colorMap.m_materialDefault = TRACE(NvFlowGetDefaultRenderMaterial(m_colorMap.m_materialPool));

	// set invalid by default for mat0 and mat1
	materialParams.material = NvFlowGridMaterialHandle{ nullptr, 1u };
	m_colorMap.m_material0 = TRACE(NvFlowCreateRenderMaterial(flowContext->m_renderContext, m_colorMap.m_materialPool, &materialParams));
	m_colorMap.m_material1 = TRACE(NvFlowCreateRenderMaterial(flowContext->m_renderContext, m_colorMap.m_materialPool, &materialParams));

	m_renderParams.materialPool = m_colorMap.m_materialPool;

	if (m_enableVolumeShadow)
	{
		NvFlowVolumeShadowDesc volumeShadowDesc = {};
		volumeShadowDesc.gridExport = gridExport;
		volumeShadowDesc.mapWidth = 4 * 256u;
		volumeShadowDesc.mapHeight = 4 * 256u;
		volumeShadowDesc.mapDepth = 4 * 256u;

		volumeShadowDesc.minResidentScale = 0.25f * (1.f / 64.f);
		volumeShadowDesc.maxResidentScale = m_shadowResidentScale * 4.f * 0.25f * (1.f / 64.f);

		m_volumeShadow = TRACE(NvFlowCreateVolumeShadow(flowContext->m_renderContext, &volumeShadowDesc));
	}
}

void FlowGridActor::release()
{
	TRACE(NvFlowReleaseGrid(m_grid));
	TRACE(NvFlowReleaseGridProxy(m_gridProxy));
	TRACE(NvFlowReleaseVolumeRender(m_volumeRender));
	TRACE(NvFlowReleaseCrossSection(m_crossSection));
	TRACE(NvFlowReleaseGridSummary(m_gridSummary));
	TRACE(NvFlowReleaseGridSummaryStateCPU(m_gridSummaryStateCPU));
	TRACE(NvFlowReleaseRenderMaterialPool(m_colorMap.m_materialPool));
	if (m_volumeShadow) TRACE(NvFlowReleaseVolumeShadow(m_volumeShadow));
	m_volumeShadow = nullptr;
}

void FlowGridActor::updatePreEmit(FlowContext* flowContext, float dt)
{
	TRACE(NvFlowGridSetParams(m_grid, &m_gridParams));
	TRACE(NvFlowGridSetMaterialParams(m_grid, NvFlowGridGetDefaultMaterial(m_grid), &m_materialParams));
	TRACE(NvFlowRenderMaterialUpdate(m_colorMap.m_materialDefault, &m_renderMaterialDefaultParams));
	TRACE(NvFlowRenderMaterialUpdate(m_colorMap.m_material0, &m_renderMaterialMat0Params));
	TRACE(NvFlowRenderMaterialUpdate(m_colorMap.m_material1, &m_renderMaterialMat1Params));

	if (m_enableTranslationTest)
	{
		m_enableTranslationTestOld = true;

		m_translationTestTime += m_translationTimeScale * dt;
		if (m_translationTestTime > 120.f) m_translationTestTime = 0.f;

		bool parity = (m_translationTestTime - floorf(m_translationTestTime)) > 0.5f;
		NvFlowFloat3 gridLocation = parity ? m_translationOffsetA : m_translationOffsetB;
		TRACE(NvFlowGridSetTargetLocation(m_grid, gridLocation));
	}
	else if(m_enableTranslationTestOld)
	{
		NvFlowFloat3 gridLocation = NvFlowFloat3{ 0.f, 0.f, 0.f };
		TRACE(NvFlowGridSetTargetLocation(m_grid, gridLocation));

		m_enableTranslationTestOld = false;
	}
}

void FlowGridActor::updatePostEmit(FlowContext* flowContext, float dt, bool shouldUpdate, bool shouldReset)
{
	if (shouldReset)
	{
		NvFlowGridResetDesc resetDesc = {};
		NvFlowGridResetDescDefaults(&resetDesc);

		resetDesc.initialLocation = m_gridDesc.initialLocation;
		resetDesc.halfSize = m_gridDesc.halfSize;

		float scale = powf(1.26f, float(m_cellSizeLogScale)) * m_cellSizeScale;
		resetDesc.halfSize.x *= scale;
		resetDesc.halfSize.y *= scale;
		resetDesc.halfSize.z *= scale;

		TRACE(NvFlowGridReset(m_grid, &resetDesc));
	}

	if (shouldUpdate)
	{
		TRACE(NvFlowGridUpdate(m_grid, flowContext->m_gridContext, dt));

		// collect stats
		if (m_grid)
		{
			auto gridExport = TRACE(NvFlowGridGetGridExport(flowContext->m_gridContext, m_grid));

			// grab for debug vis use
			if (flowContext->m_gridContext == flowContext->m_renderContext)
			{
				m_gridExportDebugVis = gridExport;
			}
			else
			{
				m_gridExportDebugVis = nullptr;
			}

			if (gridExport)
			{
				NvFlowUint* numBlockss[2] = { &m_statNumVelocityBlocks , &m_statNumDensityBlocks };
				NvFlowUint* numCellss[2] = { &m_statNumVelocityCells , &m_statNumDensityCells };
				NvFlowUint* maxBlockss[2] = { &m_statMaxVelocityBlocks , &m_statMaxDensityBlocks };
				NvFlowGridTextureChannel channels[2] = { eNvFlowGridTextureChannelVelocity, eNvFlowGridTextureChannelDensity };
				for (NvFlowUint passID = 0u; passID < 2; passID++)
				{
					NvFlowUint& numBlocks = *numBlockss[passID];
					NvFlowUint& numCells = *numCellss[passID];
					NvFlowUint& maxBlocks = *maxBlockss[passID];

					auto handle = TRACE(NvFlowGridExportGetHandle(gridExport, flowContext->m_gridContext, channels[passID]));

					NvFlowGridExportLayeredView layeredView = {};
					TRACE(NvFlowGridExportGetLayeredView(handle, &layeredView));

					m_statNumLayers = handle.numLayerViews;
					numBlocks = 0u;
					maxBlocks = layeredView.mapping.maxBlocks;

					for (NvFlowUint layerIdx = 0u; layerIdx < handle.numLayerViews; layerIdx++)
					{
						NvFlowGridExportLayerView layerView = {};
						TRACE(NvFlowGridExportGetLayerView(handle, layerIdx, &layerView));

						numBlocks += layerView.mapping.numBlocks;
					}

					numCells = layeredView.mapping.shaderParams.blockDim.w * numBlocks;
				}
			}
		}

		if (m_enableVolumeShadow && m_volumeShadow)
		{
			NvFlowVolumeShadowStats stats = {};
			TRACE(NvFlowVolumeShadowGetStats(m_volumeShadow, &stats));
			m_statVolumeShadowBlocks = stats.shadowBlocksActive;
			m_statVolumeShadowCells = stats.shadowCellsActive;
		}
		else
		{
			m_statVolumeShadowBlocks = 0u;
			m_statVolumeShadowCells = 0u;
		}

		auto gridExport = TRACE(NvFlowGridGetGridExport(flowContext->m_gridContext, m_grid));

		if (m_enableGridSummary)
		{
			NvFlowGridSummaryUpdateParams updateParams = {};
			updateParams.gridExport = gridExport;
			updateParams.stateCPU = m_gridSummaryStateCPU;

			TRACE(NvFlowGridSummaryUpdate(m_gridSummary, flowContext->m_gridContext, &updateParams));

			NvFlowUint numLayers = TRACE(NvFlowGridSummaryGetNumLayers(m_gridSummaryStateCPU));
			for (NvFlowUint layerIdx = 0u; layerIdx < numLayers; layerIdx++)
			{
				NvFlowGridSummaryResult* results = nullptr;
				NvFlowUint numResults = 0u;

				TRACE(NvFlowGridSummaryGetSummaries(m_gridSummaryStateCPU, &results, &numResults, layerIdx));

				//printf("GridSummary layer(%d) numResults(%d)", layerIdx, numResults);
			}
		}

		NvFlowGridProxyFlushParams flushParams = {};
		flushParams.gridContext = flowContext->m_gridContext;
		flushParams.gridCopyContext = flowContext->m_gridCopyContext;
		flushParams.renderCopyContext = flowContext->m_renderCopyContext;
		TRACE(NvFlowGridProxyPush(m_gridProxy, gridExport, &flushParams));
	}
}

void FlowGridActor::preDraw(FlowContext* flowContext)
{
	//auto gridView = TRACE(NvFlowGridGetGridView(m_grid, m_renderContext));

	AppGraphCtxProfileBegin(m_appctx, "UpdateGridView");

	NvFlowGridProxyFlushParams flushParams = {};
	flushParams.gridContext = flowContext->m_gridContext;
	flushParams.gridCopyContext = flowContext->m_gridCopyContext;
	flushParams.renderCopyContext = flowContext->m_renderCopyContext;
	TRACE(NvFlowGridProxyFlush(m_gridProxy, &flushParams));

	auto gridExport = TRACE(NvFlowGridProxyGetGridExport(m_gridProxy, flowContext->m_renderContext));

	AppGraphCtxProfileEnd(m_appctx, "UpdateGridView");

	// replicate render params for override
	m_renderParamsOverride = m_renderParams;

	// shadow force apply
	if (m_enableVolumeShadow && m_forceApplyShadow && !m_shadowWasForceApplied)
	{
		m_shadowWasForceApplied = true;
		m_forceIntensityCompMask = m_renderMaterialDefaultParams.intensityCompMask;
		m_forceIntensityBias = m_renderMaterialDefaultParams.intensityBias;
		m_renderMaterialDefaultParams.intensityCompMask = { 0.f, 0.f, 1.f, 0.f };
		m_renderMaterialDefaultParams.intensityBias = 0.f;
	}
	if ((!m_enableVolumeShadow || !m_forceApplyShadow) && m_shadowWasForceApplied)
	{
		m_renderMaterialDefaultParams.intensityCompMask = m_forceIntensityCompMask;
		m_renderMaterialDefaultParams.intensityBias = m_forceIntensityBias;
		m_shadowWasForceApplied = false;
	}

	// volume shadow
	if (m_enableVolumeShadow)
	{
		AppGraphCtxProfileBegin(m_appctx, "VolumeShadows");

		NvFlowVolumeShadowParams shadowParams = {};

		const float halfSize = 2.f * (0.5f * (7.5f - 5.f));
		const float center = 0.5f * (7.5f + 5.f);

		DirectX::XMMATRIX projMat = DirectX::XMMatrixPerspectiveFovLH(DirectX::XM_PI / 4.f, 1.f, center - halfSize, center + halfSize);
		DirectX::XMMATRIX viewMat = DirectX::XMMatrixMultiply(
			DirectX::XMMatrixMultiply(
				DirectX::XMMatrixScaling(0.25f, 0.25f, 0.25f),
				DirectX::XMMatrixMultiply(
					DirectX::XMMatrixRotationRollPitchYaw(0.f, DirectX::XM_PI / 4.f * m_shadowPan, 0.f),	
					DirectX::XMMatrixMultiply(
						DirectX::XMMatrixRotationRollPitchYaw(-DirectX::XM_PI / 4.f, 0.f, 0.f),
						DirectX::XMMatrixMultiply(
							DirectX::XMMatrixRotationRollPitchYaw(0.f, DirectX::XM_PI / 4.f, 0.f),
							DirectX::XMMatrixRotationAxis(DirectX::XMVectorSet(1.f, 1.f, 0.f, 1.f), -DirectX::XM_PI / 4.f * m_shadowTilt)
						)
					)
				)
			),
			DirectX::XMMatrixTranslation(0.f, 0.f, 7.f)
		);

		memcpy(&shadowParams.projectionMatrix, &projMat, sizeof(NvFlowFloat4x4));
		memcpy(&shadowParams.viewMatrix, &viewMat, sizeof(NvFlowFloat4x4));

		shadowParams.materialPool = m_renderParams.materialPool;
		shadowParams.intensityScale = m_shadowIntensityScale;
		shadowParams.minIntensity = m_shadowMinIntensity;
		shadowParams.shadowBlendCompMask = m_shadowBlendCompMask;
		shadowParams.shadowBlendBias = m_shadowBlendBias;

		shadowParams.renderChannel = m_renderParams.renderChannel;
		shadowParams.renderMode = eNvFlowVolumeRenderMode_colormap;

		TRACE(NvFlowVolumeShadowUpdate(m_volumeShadow, flowContext->m_renderContext, gridExport, &shadowParams));

		gridExport = TRACE(NvFlowVolumeShadowGetGridExport(m_volumeShadow, flowContext->m_renderContext));

		AppGraphCtxProfileEnd(m_appctx, "VolumeShadows");
	}

	if (m_separateLighting)
	{
		AppGraphCtxProfileBegin(m_appctx, "Lighting");

		NvFlowVolumeLightingParams lightingParams = {};
		lightingParams.materialPool = m_renderParams.materialPool;
		lightingParams.renderChannel = m_renderParams.renderChannel;
		lightingParams.renderMode = m_renderParams.renderMode;

		m_renderParamsOverride.renderMode = eNvFlowVolumeRenderMode_raw;

		gridExport = TRACE(NvFlowVolumeRenderLightGridExport(m_volumeRender, flowContext->m_renderContext, gridExport, &lightingParams));

		AppGraphCtxProfileEnd(m_appctx, "Lighting");
	}

	m_gridExportOverride = gridExport;
}

void FlowGridActor::draw(FlowContext* flowContext, DirectX::CXMMATRIX projection, DirectX::CXMMATRIX view)
{
	memcpy(&m_renderParamsOverride.projectionMatrix, &projection, sizeof(m_renderParamsOverride.projectionMatrix));
	memcpy(&m_renderParamsOverride.viewMatrix, &view, sizeof(m_renderParamsOverride.viewMatrix));
	m_renderParamsOverride.depthStencilView = flowContext->m_dsv;
	m_renderParamsOverride.renderTargetView = flowContext->m_rtv;

	AppGraphCtxProfileBegin(m_appctx, "Render");

	if (m_renderParams.generateDepth)
	{
		// ray march with target to composite against
		{
			auto renderParamsCopy = m_renderParamsOverride;

			// invalidate renderTargetView since it is not required here
			renderParamsCopy.renderTargetView = nullptr;

			renderParamsCopy.preColorCompositeOnly = true;
			renderParamsCopy.colorCompositeOnly = false;

			TRACE(NvFlowVolumeRenderGridExport(m_volumeRender, flowContext->m_renderContext, m_gridExportOverride, &renderParamsCopy));
		}
		// composite to target
		{
			auto renderParamsCopy = m_renderParamsOverride;

			renderParamsCopy.preColorCompositeOnly = false;
			renderParamsCopy.colorCompositeOnly = true;

			TRACE(NvFlowVolumeRenderGridExport(m_volumeRender, flowContext->m_renderContext, m_gridExportOverride, &renderParamsCopy));
		}
	}
	else
	{
		TRACE(NvFlowVolumeRenderGridExport(m_volumeRender, flowContext->m_renderContext, m_gridExportOverride, &m_renderParamsOverride));
	}

	AppGraphCtxProfileEnd(m_appctx, "Render");

	if (m_enableCrossSection)
	{
		AppGraphCtxProfileBegin(m_appctx, "CrossSection");

		if (flowContext->m_gridContext != flowContext->m_renderContext)
		{
			m_gridExportDebugVis = nullptr;
		}

		m_crossSectionParams.gridExport = m_gridExportOverride;
		m_crossSectionParams.gridExportDebugVis = m_gridExportDebugVis;

		// update parameters
		memcpy(&m_crossSectionParams.projectionMatrix, &projection, sizeof(m_crossSectionParams.projectionMatrix));
		memcpy(&m_crossSectionParams.viewMatrix, &view, sizeof(m_crossSectionParams.viewMatrix));
		m_crossSectionParams.depthStencilView = flowContext->m_dsv;
		m_crossSectionParams.renderTargetView = flowContext->m_rtv;

		float scale = powf(1.26f, float(m_cellSizeLogScale)) * m_cellSizeScale;
		m_crossSectionParams.crossSectionScale = scale * m_crossSectionScale;

		int backgroundID = int(m_crossSectionBackgroundColor);
		m_crossSectionParams.backgroundColor = { 0.f, 0.f, 0.f, 1.f };
		if (backgroundID == 1) m_crossSectionParams.backgroundColor = { 0.33f, 0.33f, 0.33f, 1.f };

		m_crossSectionParams.lineColor = { m_crossSectionLineColor.x, m_crossSectionLineColor.y, m_crossSectionLineColor.z, 1.f };

		m_crossSectionParams.materialPool = m_renderParams.materialPool;

		TRACE(NvFlowCrossSectionRender(m_crossSection, flowContext->m_renderContext, &m_crossSectionParams));

		AppGraphCtxProfileEnd(m_appctx, "CrossSection");
	}

	if (m_enableVolumeShadow && m_shadowDebugVis)
	{
		NvFlowVolumeShadowDebugRenderParams params = {};

		params.renderTargetView = flowContext->m_rtv;

		memcpy(&params.projectionMatrix, &projection, sizeof(NvFlowFloat4x4));
		memcpy(&params.viewMatrix, &view, sizeof(NvFlowFloat4x4));

		TRACE(NvFlowVolumeShadowDebugRender(m_volumeShadow, flowContext->m_renderContext, &params));
	}

	if (m_enableGridSummary && m_enableGridSummaryDebugVis)
	{
		NvFlowGridSummaryDebugRenderParams params = {};

		params.stateCPU = m_gridSummaryStateCPU;

		params.renderTargetView = flowContext->m_rtv;

		memcpy(&params.projectionMatrix, &projection, sizeof(NvFlowFloat4x4));
		memcpy(&params.viewMatrix, &view, sizeof(NvFlowFloat4x4));

		TRACE(NvFlowGridSummaryDebugRender(m_gridSummary, flowContext->m_renderContext, &params));
	}
}

// *********************** Flow Color Map *****************************************

void FlowColorMap::updateColorMap(NvFlowContext* context)
{
	NvFlowRenderMaterialHandle materials[3u] = { m_materialDefault, m_material0, m_material1 };
	std::vector<CurvePoint>* curves[3u] = {&m_curvePointsDefault, &m_curvePointsMat0, &m_curvePointsMat1 };
	for (NvFlowUint matIdx = 0u; matIdx < 3u; matIdx++)
	{
		auto& curve = *curves[matIdx];
		auto mapped = TRACE(NvFlowRenderMaterialColorMap(context, materials[matIdx]));
		if (mapped.data)
		{
			pointsToImage(mapped.data, mapped.dim, &curve[0], (unsigned int)curve.size());

			TRACE(NvFlowRenderMaterialColorUnmap(context, materials[matIdx]));
		}
	}
}

void FlowColorMap::imguiUpdate(Scene* scene, NvFlowContext* context, int border, int x, int y, int w, int h)
{
	bool actives[3u] = { m_curveEditorActiveDefault, m_curveEditorActiveMat0, m_curveEditorActiveMat1 };
	CurveEditState* editStates[3u] = { &m_editStateDefault, &m_editStateMat0 , &m_editStateMat1 };
	std::vector<CurvePoint>* curves[3u] = { &m_curvePointsDefault, &m_curvePointsMat0, &m_curvePointsMat1 };

	for (NvFlowUint matIdx = 0u; matIdx < 3u; matIdx++)
	{
		auto& editState = *editStates[matIdx];
		auto& curve = *curves[matIdx];

		if (actives[matIdx])
		{
			// curve editor
			{
				CurveEditParams params;
				params.mouseState.x = scene->m_mx;
				params.mouseState.y = scene->m_my;
				params.mouseState.but = scene->m_mbut;
				params.editorBounds.x = x + w + border;
				params.editorBounds.y = y;
				params.editorBounds.w = scene->m_winw - w - 3 * border;
				params.editorBounds.h = m_curveEditorHeight - border;
				params.rangeMin = { 0.f, 0.f, 0.f, 0.f, 0.f };
				params.rangeMax = { 1.f, 1.5f, 1.5f, 1.5f, 1.f };
				params.points = &curve[0];
				params.numPoints = (unsigned int)curve.size();

				if (curveEditor(&editState, &params))
				{
					if (editState.action == CURVE_POINT_MODIFY)
					{
						curve[editState.activePointIndex] = editState.point;
					}
					if (editState.action == CURVE_POINT_INSERT)
					{
						curve.insert(curve.begin() + editState.activePointIndex, editState.point);
					}
					if (editState.action == CURVE_POINT_REMOVE)
					{
						curve.erase(curve.begin() + editState.activePointIndex);
					}

					updateColorMap(context);
				}
			}

			break;
		}
	}

	if (imguiserOffscreenUpdate())
	{
		const char* groupNames[3u] = {"colormap", "colormapMat0", "colormapMat1"};
		for (NvFlowUint matIdx = 0u; matIdx < 3u; matIdx++)
		{
			auto& curve = *curves[matIdx];

			int oldNumItems = (int)curve.size();
			int numItems = 5 * oldNumItems;
			imguiserBeginGroup(groupNames[matIdx], &numItems);
			numItems /= 5;
			if (oldNumItems != numItems)
			{
				curve.resize(numItems);
			}

			for (size_t i = 0; i < curve.size(); i++)
			{
				auto& pt = curve[i];

				imguiserValue1f(nullptr, &pt.x);
				imguiserValue1f(nullptr, &pt.r);
				imguiserValue1f(nullptr, &pt.g);
				imguiserValue1f(nullptr, &pt.b);
				imguiserValue1f(nullptr, &pt.a);
			}

			imguiserEndGroup();
		}

		updateColorMap(context);
	}
}

bool FlowColorMap::colorMapActive(int mx, int my, unsigned char mbut)
{
	bool editorActive = m_curveEditorActiveDefault || m_curveEditorActiveMat0 || m_curveEditorActiveMat1;
	return (editorActive && my < (int)m_curveEditorHeight);
}

void FlowColorMap::initColorMap(NvFlowContext* context, const CurvePoint* pts, int numPoints, bool ptsEnabled)
{
	if (ptsEnabled)
	{
		m_curvePointsDefault.reserve(numPoints);
		m_curvePointsMat0.reserve(numPoints);
		m_curvePointsMat1.reserve(numPoints);

		m_curvePointsDefault.clear();
		m_curvePointsMat0.clear();
		m_curvePointsMat1.clear();

		for (int i = 0; i < numPoints; i++)
		{
			m_curvePointsDefault.push_back(pts[i]);
			m_curvePointsMat0.push_back(pts[i]);
			m_curvePointsMat1.push_back(pts[i]);
		}
	}

	updateColorMap(context);
}