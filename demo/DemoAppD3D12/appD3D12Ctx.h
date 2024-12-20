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

#ifndef APP_D3D12_CTX_H
#define APP_D3D12_CTX_H

#include "../DemoApp/appGraphCtx.h"

struct IDXGISwapChain3;

struct AppGraphProfilerD3D12;

struct AppDescriptorReserveHandleD3D12
{
	ID3D12DescriptorHeap* heap;
	UINT descriptorSize;
	D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle;
};

struct AppDynamicDescriptorHeapD3D12
{
	ID3D12Device* m_device = nullptr;
	ID3D12DescriptorHeap* m_heap = nullptr;
	UINT m_descriptorSize = 0u;
	UINT m_startSlot = 0u;
	UINT m_heapSize = 0u;

	void init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE heapType, UINT minHeapSize);
	void release();
	AppDescriptorReserveHandleD3D12 reserveDescriptors(UINT numDescriptors, UINT64 lastFenceCompleted, UINT64 nextFenceValue);

	AppDynamicDescriptorHeapD3D12() {}
	~AppDynamicDescriptorHeapD3D12() { release(); }
};

struct AppGraphCtxD3D12
{
	HWND                    m_hWnd = nullptr;

	int m_winW = 0;
	int m_winH = 0;
	bool m_fullscreen = false;
	bool m_valid = false;

	size_t m_dedicatedVideoMemory = 0u;

	// D3D12 non-replicated objects
	D3D12_VIEWPORT          m_viewport = {};
	D3D12_RECT              m_scissorRect = {};
	ID3D12Device*           m_device = nullptr;
	ID3D12CommandQueue*     m_commandQueue = nullptr;

	// D3D12 render target pipeline
	static const UINT       m_renderTargetCount = 6u;
	UINT                    m_renderTargetIndex = 0u;
	UINT64                  m_renderTargetID = 0u;
	IDXGISwapChain3*        m_swapChain = nullptr;
	HANDLE                  m_swapChainWaitableObject = nullptr;
	ID3D12DescriptorHeap*   m_rtvHeap = nullptr;
	UINT                    m_rtvDescriptorSize = 0u;
	ID3D12Resource*         m_renderTargets[m_renderTargetCount] = {};

	ID3D12Resource*			m_depthStencil = nullptr;
	ID3D12DescriptorHeap*	m_dsvHeap = nullptr;
	ID3D12DescriptorHeap*	m_depthSrvHeap = nullptr;

	// D3D12 frame pipeline objects
	static const UINT       m_frameCount = 8u;
	UINT                    m_frameIndex = 0u;
	UINT64                  m_frameID = 0u;
	ID3D12CommandAllocator* m_commandAllocators[m_frameCount] = {};

	// D3D12 synchronization objects
	ID3D12Fence*            m_fence = nullptr;
	HANDLE                  m_fenceEvent = 0u;
	UINT64					m_fenceValues[m_frameCount] = {};

	// fence values for library synchronization
	UINT64					m_lastFenceComplete = 1u;
	UINT64					m_thisFrameFenceID = 2u;

	// D3D12 per asset objects
	ID3D12GraphicsCommandList*	m_commandList = nullptr;
	ID3D12Resource*				m_current_renderTarget = nullptr;
	D3D12_CPU_DESCRIPTOR_HANDLE m_current_rtvHandle;
	D3D12_RENDER_TARGET_VIEW_DESC m_current_rtvDesc;
	D3D12_CPU_DESCRIPTOR_HANDLE m_current_dsvHandle;
	D3D12_DEPTH_STENCIL_VIEW_DESC m_current_dsvDesc;
	D3D12_CPU_DESCRIPTOR_HANDLE m_current_depth_srvHandle;
	D3D12_SHADER_RESOURCE_VIEW_DESC m_current_depth_srvDesc;

	DXGI_FORMAT m_rtv_format = DXGI_FORMAT_R8G8B8A8_UNORM;
	DXGI_FORMAT m_dsv_format = DXGI_FORMAT_D32_FLOAT;
	DXGI_FORMAT m_depth_srv_format = DXGI_FORMAT_R32_FLOAT;
	DXGI_FORMAT m_depth_format = DXGI_FORMAT_R32_TYPELESS;

	AppDynamicDescriptorHeapD3D12 m_dynamicHeapCbvSrvUav;

	AppGraphProfilerD3D12* m_profiler = nullptr;

	AppGraphCtxD3D12();
	~AppGraphCtxD3D12();
};

inline AppGraphCtxD3D12* cast_to_AppGraphCtxD3D12(AppGraphCtx* appctx)
{
	return (AppGraphCtxD3D12*)(appctx);
}

inline AppGraphCtx* cast_from_AppGraphCtxD3D12(AppGraphCtxD3D12* appctx)
{
	return (AppGraphCtx*)(appctx);
}

APP_GRAPH_CTX_API AppGraphCtx* AppGraphCtxCreateD3D12(int deviceID);

APP_GRAPH_CTX_API bool AppGraphCtxUpdateSizeD3D12(AppGraphCtx* context, SDL_Window* window, bool fullscreen);

APP_GRAPH_CTX_API void AppGraphCtxReleaseRenderTargetD3D12(AppGraphCtx* context);

APP_GRAPH_CTX_API void AppGraphCtxReleaseD3D12(AppGraphCtx* context);

APP_GRAPH_CTX_API void AppGraphCtxFrameStartD3D12(AppGraphCtx* context, AppGraphColor clearColor);

APP_GRAPH_CTX_API void AppGraphCtxFramePresentD3D12(AppGraphCtx* context, bool fullsync);

APP_GRAPH_CTX_API void AppGraphCtxWaitForFramesD3D12(AppGraphCtx* context, unsigned int maxFramesInFlight);

APP_GRAPH_CTX_API void AppGraphCtxProfileEnableD3D12(AppGraphCtx* context, bool enabled);

APP_GRAPH_CTX_API void AppGraphCtxProfileBeginD3D12(AppGraphCtx* context, const char* label);

APP_GRAPH_CTX_API void AppGraphCtxProfileEndD3D12(AppGraphCtx* context, const char* label);

APP_GRAPH_CTX_API bool AppGraphCtxProfileGetD3D12(AppGraphCtx* context, const char** plabel, float* cpuTime, float* gpuTime, int index);

APP_GRAPH_CTX_API size_t AppGraphCtxDedicatedVideoMemoryD3D12(AppGraphCtx* context);


APP_GRAPH_CTX_API void AppGraphCtxBeginMarkerD3D12(AppGraphCtx* context, const char* name);

APP_GRAPH_CTX_API void AppGraphCtxEndMarkerD3D12(AppGraphCtx* context, const char* name);

#endif