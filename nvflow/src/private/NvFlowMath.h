#ifndef NVFLOWMATH_H
#define NVFLOWMATH_H
#include <nvflow/NvFlowTypes.h>

namespace NvFlow {

///
/// Vector math
///

inline NvFlowDim operator/(NvFlowDim v1, NvFlowDim v2) {
    return NvFlowDim{v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};
}

inline NvFlowDim operator/(NvFlowDim v, NvFlowUint s) {
    return NvFlowDim{v.x / s, v.y / s, v.z / s};
}

inline NvFlowDim operator+(NvFlowDim v1, NvFlowDim v2) {
    return NvFlowDim{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
}

inline NvFlowDim operator+(NvFlowDim v, NvFlowUint s) {
    return NvFlowDim{v.x + s, v.y + s, v.z + s};
}

inline NvFlowDim operator+(NvFlowUint s, NvFlowDim v) {
    return NvFlowDim{v.x + s, v.y + s, v.z + s};
}

}  // namespace NvFlow

#endif /* NVFLOWMATH_H */
