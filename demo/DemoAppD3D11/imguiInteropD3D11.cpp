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

//direct3d headers
#include <d3d11_1.h>

// include the Direct3D Library file
#pragma comment (lib, "d3d11.lib")

#include "../DemoApp/imguiInterop.h"

#include "appD3D11Ctx.h"
#include "imguiGraphD3D11.h"

IMGUI_GRAPH_API bool imguiInteropGraphInitD3D11(imguiGraphInit_t func, const char* fontpath, AppGraphCtx* appctx);

IMGUI_GRAPH_API void imguiInteropGraphUpdateD3D11(imguiGraphUpdate_t func, AppGraphCtx* appctx);

bool imguiInteropGraphInitD3D11(imguiGraphInit_t func, const char* fontpath, AppGraphCtx* appctxIn)
{
	auto appctx = cast_to_AppGraphCtxD3D11(appctxIn);

	ImguiGraphDescD3D11 desc;
	desc.device = appctx->m_device;
	desc.deviceContext = appctx->m_deviceContext;
	desc.winW = appctx->m_winW;
	desc.winH = appctx->m_winH;

	return func(fontpath, cast_from_imguiGraphDescD3D11(&desc));
}

void imguiInteropGraphUpdateD3D11(imguiGraphUpdate_t func, AppGraphCtx* appctxIn)
{
	auto appctx = cast_to_AppGraphCtxD3D11(appctxIn);

	ImguiGraphDescD3D11 desc;
	desc.device = appctx->m_device;
	desc.deviceContext = appctx->m_deviceContext;
	desc.winW = appctx->m_winW;
	desc.winH = appctx->m_winH;

	return func(cast_from_imguiGraphDescD3D11(&desc));
}