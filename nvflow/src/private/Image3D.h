#ifndef IMAGE3D_H
#define IMAGE3D_H
#include "Types.h"
#include <vector>

namespace NvFlow {

template <typename T>
struct Image3D {
    Image3D();
    Image3D(NvFlowDim dim);

    T& operator[](ptrdiff_t i);

    const T& operator[](ptrdiff_t i) const;

    void resize(NvFlowDim dim, const T& val = T{});

    size_t dim1() const;

    NvFlowDim dim() const;

    T* data();

    const T* data() const;

    const T& operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const;

    T& operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k);

 private:
    NvFlowDim m_dim;
    std::vector<T> m_data;
};
template <typename T>
inline Image3D<T>::Image3D() : m_dim{0, 0, 0} {
}
template <typename T>
inline Image3D<T>::Image3D(NvFlowDim dim) : Image3D() {
    resize(dim);
}

template <typename T>
inline T& Image3D<T>::operator[](ptrdiff_t i) {
    return m_data[i];
}

template <typename T>
inline const T& Image3D<T>::operator[](ptrdiff_t i) const {
    return m_data[i];
}

template <typename T>
inline void Image3D<T>::resize(NvFlowDim dim, const T& val) {
    m_dim = dim;
    size_t size1 = dim.x;
    size1 *= dim.y * dim.z;
    m_data.resize(size1, val);
}

template <typename T>
inline size_t Image3D<T>::dim1() const {
    return m_data.size();
}

template <typename T>
inline NvFlowDim Image3D<T>::dim() const {
    return m_dim;
}

template <typename T>
inline T* Image3D<T>::data() {
    return m_data.data();
}

template <typename T>
inline const T* Image3D<T>::data() const {
    return m_data.data();
}

template <typename T>
inline const T& Image3D<T>::operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const {
    return m_data[i + m_dim.x * (j + k * m_dim.y)];
}

template <typename T>
inline T& Image3D<T>::operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) {
    return m_data[i + m_dim.x * (j + k * m_dim.y)];
}

};  // namespace NvFlow

#endif /* IMAGE3D_H */
