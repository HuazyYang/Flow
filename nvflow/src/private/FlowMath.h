#ifndef NVFLOWMATH_H
#define NVFLOWMATH_H
#include <nvflow/NvFlowTypes.h>
#include "BasicMath.h"

namespace NvFlow {

///
/// Scalar math
///
inline float asfloat(int x) {
    return *reinterpret_cast<float *>(&x);
}

inline float asfloat(unsigned int x) {
    return *reinterpret_cast<float *>(&x);
}

inline int asint(float x) {
    return *reinterpret_cast<int *>(&x);
}

inline unsigned int asuint(float x) {
    return *reinterpret_cast<unsigned int *>(&x);
}

///
/// Vector math
///

#define NVFLOW_VECTOR3_OPERATORS(VecType, CompType, FuncSuffix)               \
    inline VecType make_##FuncSuffix(CompType s) {                            \
        return VecType{s, s, s};                                              \
    }                                                                         \
    inline VecType make_##FuncSuffix(CompType s1, CompType s2, CompType s3) { \
        return VecType{s1, s2, s3};                                           \
    }                                                                         \
    template <typename Ty>                                                    \
    inline VecType make_##FuncSuffix(Ty s) {                                  \
        CompType s1 = static_cast<CompType>(s);                               \
        return VecType{s1, s1, s1};                                           \
    }                                                                         \
    template <typename Ty>                                                    \
    inline VecType make_##FuncSuffix(Ty s1, Ty s2, Ty s3) {                   \
        return VecType{static_cast<CompType>(s1), static_cast<CompType>(s2),  \
                       static_cast<CompType>(s3)};                            \
    }                                                                         \
    inline bool operator==(const VecType &left, const VecType &right) {       \
        return left.x == right.x && left.y == right.y && left.z == right.z;   \
    }                                                                         \
    inline bool operator!=(const VecType &left, const VecType &right) {       \
        return left.x != right.x || left.y != right.y || left.z != right.z;   \
    }                                                                         \
    inline VecType operator/(VecType v1, VecType v2) {                        \
        return VecType{v1.x / v2.x, v1.y / v2.y, v1.z / v2.z};                \
    }                                                                         \
                                                                              \
    inline VecType operator/(VecType v, CompType s) {                         \
        return VecType{v.x / s, v.y / s, v.z / s};                            \
    }                                                                         \
    inline VecType operator/(CompType s, VecType v) {                         \
        return VecType{s / v.x, s / v.y, s / v.z};                            \
    }                                                                         \
    inline VecType operator+(VecType v1, VecType v2) {                        \
        return VecType{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};                \
    }                                                                         \
    inline VecType operator+(VecType v, CompType s) {                         \
        return VecType{v.x + s, v.y + s, v.z + s};                            \
    }                                                                         \
    inline VecType operator+(CompType s, VecType v) {                         \
        return VecType{s + v.x, s + v.y, s + v.z};                            \
    }                                                                         \
    inline VecType operator-(VecType v1, VecType v2) {                        \
        return VecType{v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};                \
    }                                                                         \
    inline VecType operator-(VecType v, CompType s) {                         \
        return VecType{v.x - s, v.y - s, v.z - s};                            \
    }                                                                         \
    inline VecType operator-(CompType s, VecType v) {                         \
        return VecType{s - v.x, s - v.y, s - v.z};                            \
    }                                                                         \
    inline VecType operator*(VecType v1, VecType v2) {                        \
        return VecType{v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};                \
    }                                                                         \
                                                                              \
    inline VecType operator*(VecType v, CompType s) {                         \
        return VecType{v.x * s, v.y * s, v.z * s};                            \
    }                                                                         \
                                                                              \
    inline VecType operator*(CompType s, VecType v) {                         \
        return VecType{v.x * s, v.y * s, v.z * s};                            \
    }

#define NVFLOW_VECTOR3_BIT_OPERATORS(VecType)              \
    inline VecType operator>>(VecType v, unsigned int s) { \
        return VecType{v.x >> s, v.y >> s, v.z >> s};      \
    }

#define NVFLOW_VECTOR4_OPERATORS(VecType, CompType, FuncSuffix)                            \
    inline VecType make_##FuncSuffix(CompType s) {                                         \
        return VecType{s, s, s, s};                                                        \
    }                                                                                      \
    inline VecType make_##FuncSuffix(CompType s1, CompType s2, CompType s3, CompType s4) { \
        return VecType{s1, s2, s3, s4};                                                    \
    }                                                                                      \
    template <typename Ty>                                                                 \
    inline VecType make_##FuncSuffix(Ty s) {                                               \
        CompType s1 = static_cast<CompType>(s);                                            \
        return VecType{s1, s1, s1, s1};                                                    \
    }                                                                                      \
    template <typename Ty>                                                                 \
    inline VecType make_##FuncSuffix(Ty s1, Ty s2, Ty s3, Ty s4) {                         \
        return VecType{static_cast<CompType>(s1), static_cast<CompType>(s2),               \
                       static_cast<CompType>(s3), static_cast<CompType>(s4)};              \
    }                                                                                      \
    inline bool operator==(const VecType &left, const VecType &right) {                    \
        return left.x == right.x && left.y == right.y && left.z == right.z &&              \
               left.w == right.w;                                                          \
    }                                                                                      \
                                                                                           \
    inline VecType operator/(VecType v1, VecType v2) {                                     \
        return VecType{v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w};                \
    }                                                                                      \
                                                                                           \
    inline VecType operator/(VecType v, CompType s) {                                      \
        return VecType{v.x / s, v.y / s, v.z / s, v.w / s};                                \
    }                                                                                      \
    inline VecType operator/(CompType s, VecType v) {                                      \
        return VecType{s / v.x, s / v.y, s / v.z, s / v.w};                                \
    }                                                                                      \
    inline VecType operator+(VecType v1, VecType v2) {                                     \
        return VecType{v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w};                \
    }                                                                                      \
                                                                                           \
    inline VecType operator+(VecType v, CompType s) {                                      \
        return VecType{v.x + s, v.y + s, v.z + s, v.w + s};                                \
    }                                                                                      \
                                                                                           \
    inline VecType operator+(CompType s, VecType v) {                                      \
        return VecType{v.x + s, v.y + s, v.z + s, v.w + s};                                \
    }                                                                                      \
    inline VecType operator-(VecType v1, VecType v2) {                                     \
        return VecType{v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w};                \
    }                                                                                      \
                                                                                           \
    inline VecType operator-(VecType v, CompType s) {                                      \
        return VecType{v.x - s, v.y - s, v.z - s, v.w - s};                                \
    }                                                                                      \
                                                                                           \
    inline VecType operator-(CompType s, VecType v) {                                      \
        return VecType{s - v.x, s - v.y, s - v.z, s - v.w};                                \
    }                                                                                      \
    inline VecType operator*(VecType v1, VecType v2) {                                     \
        return VecType{v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w};                \
    }                                                                                      \
                                                                                           \
    inline VecType operator*(VecType v, CompType s) {                                      \
        return VecType{v.x * s, v.y * s, v.z * s, v.w * s};                                \
    }                                                                                      \
                                                                                           \
    inline VecType operator*(CompType s, VecType v) {                                      \
        return VecType{v.x * s, v.y * s, v.z * s, v.w * s};                                \
    }

#define NVFLOW_VECTOR4_BIT_OPERATORS(VecType)                   \
    inline VecType operator>>(VecType v, unsigned int s) {      \
        return VecType{v.x >> s, v.y >> s, v.z >> s, v.w >> s}; \
    }

NVFLOW_VECTOR3_OPERATORS(NvFlowDim, NvFlowUint, dim)
NVFLOW_VECTOR3_BIT_OPERATORS(NvFlowDim)

NVFLOW_VECTOR3_OPERATORS(NvFlowUint3, NvFlowUint, uint3)
NVFLOW_VECTOR3_BIT_OPERATORS(NvFlowUint3)

NVFLOW_VECTOR3_OPERATORS(NvFlowInt3, int, int3)
NVFLOW_VECTOR3_BIT_OPERATORS(NvFlowInt3)

NVFLOW_VECTOR3_OPERATORS(NvFlowFloat3, float, float3)

NVFLOW_VECTOR4_OPERATORS(NvFlowUint4, NvFlowUint, uint4)
NVFLOW_VECTOR4_BIT_OPERATORS(NvFlowUint4)

NVFLOW_VECTOR4_OPERATORS(NvFlowInt4, int, int4)
NVFLOW_VECTOR4_BIT_OPERATORS(NvFlowInt4)

NVFLOW_VECTOR4_OPERATORS(NvFlowFloat4, float, float4)

inline NvFlowDim make_dim(const NvFlowUint3 &v) {
    return NvFlowDim{v.x, v.y, v.z};
}

inline NvFlowDim make_dim(const NvFlowUint4 &v) {
    return NvFlowDim{v.x, v.y, v.z};
}

inline NvFlowUint3 make_uint3(const NvFlowDim &v) {
    return NvFlowUint3{v.x, v.y, v.z};
}

inline NvFlowInt4 make_int4(const NvFlowInt3 &v, int s) {
    return NvFlowInt4{v.x, v.y, v.z, s};
}

inline NvFlowInt4 make_int4(const NvFlowFloat4 &v) {
    return NvFlowInt4{int(v.x), int(v.y), int(v.z), int(v.w)};
}

inline NvFlowInt4 make_int4(const NvFlowFloat3 &v, float s) {
    return NvFlowInt4{int(v.x), int(v.y), int(v.z), int(s)};
}

inline NvFlowUint4 make_uint4(const NvFlowUint3 &v, NvFlowUint s) {
    return NvFlowUint4{v.x, v.y, v.z, s};
}

inline NvFlowUint4 make_uint4(const NvFlowDim dim, NvFlowUint s) {
    return NvFlowUint4{dim.x, dim.y, dim.z, s};
}

inline NvFlowFloat3 make_float3(const NvFlowDim &v) {
    return NvFlowFloat3{float(v.x), float(v.y), float(v.z)};
}

inline NvFlowUint4 make_uint4(const NvFlowFloat4 &v) {
    return NvFlowUint4{(unsigned int)(v.x), (unsigned int)(v.y), (unsigned int)(v.z),
                       (unsigned int)(v.w)};
}

inline NvFlowFloat4 make_float4(const NvFlowFloat3 &v, float s) {
    return NvFlowFloat4{v.x, v.y, v.z, s};
}

inline NvFlowFloat4 make_float4(const NvFlowDim &v, NvFlowUint s) {
    return NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(s)};
}

inline NvFlowFloat4 make_float4(const NvFlowUint3 &v, NvFlowUint s) {
    return NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(s)};
}

inline NvFlowFloat4 make_float4(const NvFlowUint4 &v) {
    return NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(v.w)};
}

inline NvFlowFloat4 make_float4(const NvFlowInt4 &v) {
    return NvFlowFloat4{float(v.x), float(v.y), float(v.z), float(v.w)};
}

inline NvFlowInt3 make_int3(const NvFlowFloat3 &v) {
    return NvFlowInt3{int(v.x), int(v.y), int(v.z)};
}

inline NvFlowFloat3 make_float3(const NvFlowInt3 &v) {
    return NvFlowFloat3{float(v.x), float(v.y), float(v.z)};
}

inline NvFlowFloat3 make_float3(const NvFlowUint3 &v) {
    return NvFlowFloat3{float(v.x), float(v.y), float(v.z)};
}

inline NvFlowFloat3 make_float3(const NvFlowFloat4 &v) {
    return NvFlowFloat3{v.x, v.y, v.z};
}

inline unsigned int log2ui(unsigned int x) {
    for (int i = 0; i < 32; ++i) {
        if ((1 << i) >= x)
            return i;
    }
    return 0;
}

template <typename T>
inline T max3(T x, T y, T z) noexcept {
    return max(x, max(y, z));
}

NvFlowFloat3 normalize(const NvFlowFloat3 &v);

inline NvFlowFloat3 abs(const NvFlowFloat3 &v) noexcept {
    return NvFlowFloat3{abs(v.x), abs(v.y), abs(v.z)};
}

inline NvFlowFloat4 abs(const NvFlowFloat4 &v) noexcept {
    return NvFlowFloat4{abs(v.x), abs(v.y), abs(v.z), abs(v.w)};
}

inline NvFlowFloat3 min(const NvFlowFloat3 &v1, const NvFlowFloat3 &v2) noexcept {
    return NvFlowFloat3{min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z)};
}

inline NvFlowFloat4 min(const NvFlowFloat4 &v1, const NvFlowFloat4 &v2) noexcept {
    return NvFlowFloat4{min(v1.x, v2.x), min(v1.y, v2.y), min(v1.z, v2.z), min(v1.w, v2.w)};
}

inline NvFlowFloat3 max(const NvFlowFloat3 &v1, const NvFlowFloat3 &v2) noexcept {
    return NvFlowFloat3{max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z)};
}

inline NvFlowFloat4 max(const NvFlowFloat4 &v1, const NvFlowFloat4 &v2) noexcept {
    return NvFlowFloat4{max(v1.x, v2.x), max(v1.y, v2.y), max(v1.z, v2.z), max(v1.w, v2.w)};
}

inline NvFlowFloat4 floor(const NvFlowFloat4 &v) noexcept {
    return NvFlowFloat4{floor(v.x), floor(v.y), floor(v.z), floor(v.w)};
}

inline NvFlowFloat4 ceil(const NvFlowFloat4 &v) noexcept {
    return NvFlowFloat4{ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w)};
}

NvFlowFloat4 pow(const NvFlowFloat4 &a, const NvFlowFloat4 &b) noexcept;

NvFlowFloat4 pow(const NvFlowFloat4 &a, float b) noexcept;

float dot(const NvFlowFloat3 &a, const NvFlowFloat3 &b) noexcept;

inline float lerp(float a, float b, float t) noexcept {
    return (1.f - t) * a + t * b;
}

inline NvFlowFloat3 lerp(const NvFlowFloat3 a, const NvFlowFloat3 &b, float t) noexcept {
    return (1.f - t) * a + t * b;
}

// Matrix

NvFlowFloat4x4 transpose(const NvFlowFloat4x4 &m) noexcept;

float vector3Length(const NvFlowFloat4 &v) noexcept;

NvFlowFloat4 transform4(const NvFlowFloat4 &v, const NvFlowFloat4x4 &m) noexcept;

NvFlowFloat4x4 matrixNormalize(const NvFlowFloat4x4 &m) noexcept;

NvFlowFloat4x4 inverse(const NvFlowFloat4x4 &m) noexcept;

NvFlowFloat4x4 identity() noexcept;

NvFlowFloat4x4 operator*(const NvFlowFloat4x4 &a, const NvFlowFloat4x4 &b) noexcept;

NvFlowFloat4x4 matrixTranslation(float x, float y, float z);

NvFlowFloat4x4 matrixScaling(float x, float y, float z);

}  // namespace NvFlow

#endif /* NVFLOWMATH_H */
