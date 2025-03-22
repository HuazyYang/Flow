#include <cstdint>
#include "FlowMath.h"
#include <cstring>
#include <DirectXMath.h>

namespace NvFlow {

using namespace DirectX;

inline XMVECTOR xm_cast(const NvFlowFloat3& v) {
    return XMLoadFloat3((const XMFLOAT3*)&v);
}

inline XMVECTOR xm_cast(const NvFlowFloat4& v) {
    return XMLoadFloat4((const XMFLOAT4*)&v);
}

inline XMMATRIX xm_cast(const NvFlowFloat4x4& m) {
    return XMLoadFloat4x4((const XMFLOAT4X4*)&m);
}

template <typename Ty>
Ty xm_rcast(XMVECTOR v);

template <>
inline NvFlowFloat3 xm_rcast<NvFlowFloat3>(XMVECTOR v) {
    NvFlowFloat3 r;
    XMStoreFloat3((XMFLOAT3*)&r, v);
    return r;
}

template <>
inline NvFlowFloat4 xm_rcast<NvFlowFloat4>(XMVECTOR v) {
    NvFlowFloat4 r;
    XMStoreFloat4((XMFLOAT4*)&r, v);
    return r;
}

inline NvFlowFloat4x4 xm_rcast(XMMATRIX M) {
    NvFlowFloat4x4 r;
    XMStoreFloat4x4((XMFLOAT4X4*)&r, M);
    return r;
}

NvFlowFloat3 normalize(const NvFlowFloat3& v) {
    XMVECTOR V = xm_cast(v);
    V = XMVector3Normalize(V);
    return xm_rcast<NvFlowFloat3>(V);
}

NvFlowFloat4 pow(const NvFlowFloat4& a, const NvFlowFloat4& b) noexcept {
    XMVECTOR A = xm_cast(a);
    XMVECTOR B = xm_cast(b);
    XMVECTOR C = XMVectorPow(A, B);
    return xm_rcast<NvFlowFloat4>(C);
}

NvFlowFloat4 pow(const NvFlowFloat4& a, float b) noexcept {
    XMVECTOR A = xm_cast(a);
    XMVECTOR B = XMVectorReplicate(b);
    XMVECTOR C = XMVectorPow(A, B);
    return xm_rcast<NvFlowFloat4>(C);
}

float dot(const NvFlowFloat3& a, const NvFlowFloat3& b) noexcept {
    XMVECTOR A = xm_cast(a);
    XMVECTOR B = xm_cast(b);
    XMVECTOR R = XMVector3Dot(A, B);
    return XMVectorGetX(R);
}

NvFlowFloat4x4 transpose(const NvFlowFloat4x4& m) noexcept {
    XMMATRIX M = XMLoadFloat4x4((const XMFLOAT4X4*)&m);
    M = XMMatrixTranspose(M);
    NvFlowFloat4x4 result;
    XMStoreFloat4x4((XMFLOAT4X4*)&result, M);
    return result;
}

float vector3Length(const NvFlowFloat4& v) noexcept {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

NvFlowFloat4 transform4(const NvFlowFloat4& v, const NvFlowFloat4x4& m) noexcept {
    NvFlowFloat4 result;
    XMVECTOR V = XMLoadFloat4((const XMFLOAT4*)&v);
    XMMATRIX M = XMLoadFloat4x4((const XMFLOAT4X4*)&m);
    XMVECTOR Result = XMVector4Transform(V, M);
    XMStoreFloat4((XMFLOAT4*)&result, Result);
    return result;
}

NvFlowFloat4x4 matrixNormalize(const NvFlowFloat4x4& m) noexcept {
    NvFlowFloat4x4 result;
    XMVECTOR V0, V[4];
    V0 = XMLoadFloat4((const XMFLOAT4*)&m.x);
    XMVectorSetW(V0, 0.f);
    V[0] = XMVector4Normalize(V0);
    V0 = XMLoadFloat4((const XMFLOAT4*)&m.y);
    XMVectorSetW(V0, 0.f);
    V[1] = XMVector4Normalize(V0);
    V0 = XMLoadFloat4((const XMFLOAT4*)&m.z);
    XMVectorSetW(V0, 0.f);
    V[2] = XMVector4Normalize(V0);

    XMStoreFloat4((XMFLOAT4*)&result.x, V[0]);
    XMStoreFloat4((XMFLOAT4*)&result.y, V[1]);
    XMStoreFloat4((XMFLOAT4*)&result.z, V[2]);
    result.w = NvFlowFloat4{0.f, 0.f, 0.f, 1.f};

    return result;
}

NvFlowFloat4x4 inverse(const NvFlowFloat4x4& m) noexcept {
    NvFlowFloat4x4 r;
    XMMATRIX M = XMLoadFloat4x4((const XMFLOAT4X4*)&m);
    M = XMMatrixInverse(nullptr, M);
    XMStoreFloat4x4((XMFLOAT4X4*)&r, M);
    return r;
}

NvFlowFloat4x4 identity() noexcept {
    NvFlowFloat4x4 result;
    XMMATRIX I = XMMatrixIdentity();
    XMStoreFloat4x4((XMFLOAT4X4*)&result, I);
    return result;
}

NvFlowFloat4x4 operator*(const NvFlowFloat4x4& a, const NvFlowFloat4x4& b) noexcept {
    NvFlowFloat4x4 r;
    XMMATRIX A = XMLoadFloat4x4((const XMFLOAT4X4*)&a);
    XMMATRIX B = XMLoadFloat4x4((const XMFLOAT4X4*)&b);
    XMMATRIX R = XMMatrixMultiply(A, B);
    XMStoreFloat4x4((XMFLOAT4X4*)&r, R);
    return r;
}

NvFlowFloat4x4 matrixTranslation(float x, float y, float z) {
    XMMATRIX T = XMMatrixTranslation(x, y, z);
    return xm_rcast(T);
}

NvFlowFloat4x4 matrixScaling(float x, float y, float z) {
    XMMATRIX S = XMMatrixScaling(x, y, z);
    return xm_rcast(S);
}

}  // namespace NvFlow