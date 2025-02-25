#ifndef TYPES_H
#define TYPES_H
#include <nvflow/NvFlowTypes.h>
#include <cstdint>
#include <dxgi.h>
#include <type_traits>
#include <cassert>

namespace NvFlow {

using BYTE = uint8_t;

template <typename T, size_t N>
char (*countofHelper(const T (&Array)[N]))[N];

#define countof(Array) sizeof(*countofHelper(Array))

// Constants
constexpr float FLOAT_0_5 = 0.5f;
constexpr float FLOAT_1_0 = 1.f;
constexpr float FLOAT_1_5 = 1.5f;

uint32_t getFormatSizeInBytes(NvFlowFormat format);

DXGI_FORMAT convertToDXGI(NvFlowFormat format);

NvFlowFormat convertToNvFlow(DXGI_FORMAT format);

template <int N, typename T>
T alignUp(T ptr) {
    static_assert((N & (N - 1)) == 0, "alignUp dividend must be power of 2");
    constexpr T mask = ~T(N - 1);
    return (ptr + T(N - 1)) & mask;
}

NvFlowDim getTileDim(NvFlowFormat format);

///
/// Error handling
///

#if _MSC_VER
#define NVFLOW_FUNCSIG __FUNCSIG__
#else
#define NVFLOW_FUNCSIG __PRETTY_FUNCTION__
#endif

enum CommonErrorType {
    COMMON_ERROR_UNKNOWN = 0,
    COMMON_ERROR_NOT_IMPLEMENTED = 1,
    COMMON_ERROR_INDEX_OUT_OF_RANGE = 2,
};

#define NVFLOW_NOT_IMPLEMENTED_ERROR()                      \
    NvFlow::HandleError(__FILE__, __LINE__, NVFLOW_FUNCSIG, \
                        NvFlow::COMMON_ERROR_NOT_IMPLEMENTED)

#define NVFLOW_INDEX_OUT_OF_RANGE_ERROR()                   \
    NvFlow::HandleError(__FILE__, __LINE__, NVFLOW_FUNCSIG, \
                        NvFlow::COMMON_ERROR_INDEX_OUT_OF_RANGE)

#define NVFLOW_ASSERT(expr) assert(expr)

void HandleError(const char* file, uint32_t line, const char* pos, CommonErrorType errType);

}  // namespace NvFlow

#include "NvFlowMath.h"

#endif /* TYPES_H */
