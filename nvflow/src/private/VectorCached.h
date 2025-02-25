#ifndef VECTORCACHED_H
#define VECTORCACHED_H
#include "Types.h"
#include "type_traits"
#include "Allocable.h"
#include <limits>

namespace NvFlow {

template <typename T, uint32_t N>
class VectorCached {
 public:
    using size_type = uint32_t;
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    static constexpr size_type cached_size = N;

    // Type sanity check
    /*     static_assert(std::is_trivially_default_constructible_v<value_type> &&
                          std::is_trivially_destructible_v<value_type> &&
                          std::is_trivially_copy_constructible_v<value_type> &&
                          std::is_trivially_move_constructible_v<value_type>,
                      "None-POD type can not instantiate VectorCached template"); */

    VectorCached(size_type capacity = cached_size) {
        m_size = 0;
        m_capacity = capacity;
        m_data = allocate(capacity);

        for (size_type i = 0; i < capacity; ++i)
            ::new (m_data + i) value_type();
    }

    ~VectorCached() {
        cleanup(m_data, m_capacity);
        m_data = nullptr;
        m_capacity = 0;
        m_size = 0;
    }

    VectorCached(const VectorCached &) = delete;
    VectorCached &operator=(const VectorCached &other) = delete;

    reference operator[](size_type idx) {
        check_range(idx);
        return m_data[idx];
    }

    value_type *data() { return m_data; }

    const value_type *data() const { return m_data; }

    size_type allocateBack() {
        reserve(m_size + 1);
        return m_size++;
    }

    void resize(size_type size) {
        reserve(size);
        m_size = size;
    }

    size_type size() const { return m_size; }

    void push_back(const T &val) {
        auto idx = allocateBack();
        m_data[idx] = val;
    }

    void push_back(T &&val) {
        auto idx = allocateBack();
        m_data[idx] = std::forward<T>(val);
    }

    T &back() {
        check_range(0);
        return m_data[m_size - 1];
    }

    const T &back() const {
        check_range(0);
        return m_data[m_size - 1];
    }

    VectorCached(VectorCached *rhs) {
        m_data = rhs->m_data;
        m_capacity = rhs->m_capacity;
        m_size = rhs->m_size;
        if ((uint8_t *)rhs->m_data == rhs->m_cache) {
            m_data = (value_type *)m_cache;
            for (size_type i = 0; i < rhs->m_capacity; ++i)
                m_data[i] = rhs->m_data[i];
        }
        rhs->m_data = rhs->m_cache;
        rhs->m_capacity = cached_size;
        rhs->m_size = 0;
    }

    void reserve(size_type requestedCapacity) {
        size_type capacity;
        for (capacity = m_capacity; capacity < requestedCapacity; capacity *= 2)
            ;

        if (capacity > m_capacity) {
            auto newData = (value_type *)Allocable::allocate(capacity);

            if (newData != m_data) {
                for (size_type i = 0; i < m_size; ++i) {
                    auto newVal = ::new (newData + i) value_type();
                    auto oldVal = &m_data[i];
                    *newVal = std::move(*oldVal);
                }
            }

            for (size_type i = m_size; i < capacity; ++i)
                ::new (newData + i) value_type();

            if (newData != m_data) cleanup(m_data, m_capacity);
            m_data = newData;
            m_capacity = capacity;
        }
    }

 private:
    void check_range(size_type idx) const {
        if (idx >= m_size) NVFLOW_INDEX_OUT_OF_RANGE_ERROR();
    }

    value_type *allocate(size_type capacity) {
        if (capacity > cached_size)
            return (value_type *)Allocable::allocate(sizeof(value_type) * capacity);
        else
            return (value_type *)m_cache;
    }

    void cleanup(T *data, size_type capacity) {
        for (size_t i = 0; i < capacity; ++i)
            data[i].~value_type();
        if (data != (T *)m_cache) Allocable::deallocate(data);
    }

    value_type *m_data;
    size_type m_capacity;
    size_type m_size;
    alignas(alignof(T)) uint8_t m_cache[sizeof(value_type) * cached_size];
};

template <typename T, uint32_t M, uint32_t N>
class VectorCached2D {
 public:
    using size_type = uint32_t;
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T &;
    using const_reference = const T &;
    static constexpr size_type cached_size_x = M;
    static constexpr size_type cached_size_y = N;
    static constexpr size_type cached_size = cached_size_x * cached_size_y;

    // Type sanity check
    static_assert(std::is_trivially_default_constructible_v<value_type> &&
                      std::is_trivially_destructible_v<value_type> &&
                      std::is_trivially_copy_constructible_v<value_type> &&
                      std::is_trivially_move_constructible_v<value_type>,
                  "None-POD type can not instantiate VectorCached template");

    struct Row {
        value_type *m_data;
        size_type m_size;

        reference operator[](size_type idx) {
            check_range(idx);
            return m_data[idx];
        }

     private:
        void check_range(size_type idx) const {
            if (idx >= m_size) NVFLOW_INDEX_OUT_OF_RANGE_ERROR();
        }
    };

    VectorCached2D(size_type capacityX = M, size_type capacityY = N) {
        m_sizeX = 0;
        m_sizeY = 0;
        m_capacityX = capacityX;
        m_capacityY = capacityY;
        m_data = allocate(capacityX, capacityY);
        size_type capacity = capacityX * capacityY;

        for (size_type i = 0; i < capacity; ++i)
            ::new (m_data + i) value_type();
    }

    ~VectorCached2D() {
        cleanup(m_data, m_capacityX, m_capacityY);
        m_data = nullptr;
        m_capacityX = 0;
        m_capacityY = 0;
        m_sizeX = 0;
        m_sizeY = 0;
    }

    VectorCached2D(const VectorCached2D &) = delete;
    VectorCached2D &operator=(const VectorCached2D &other) = delete;

    Row operator[](size_type idx) {
        check_range(idx);
        return Row{m_data + m_capacityX * idx, m_sizeX};
    }

    size_type allocateBackX() {
        reserve(m_sizeX + 1, m_sizeY);
        return m_sizeX++;
    }

    size_type allocateBackY() {
        reserve(m_sizeX, m_sizeY + 1);
        return m_sizeY++;
    }

    void resize(size_type sizeX, size_type sizeY) {
        reserve(sizeX, sizeY);
        m_sizeX = sizeX;
        m_sizeY = sizeY;
    }

    size_type sizeX() const { return m_sizeX; }

    size_type sizeY() const { return m_sizeY; }

    void reserve(size_type requestedCapacityX, size_type requestedCapacityY) {
        size_type capacityX = m_capacityX, capacityY = m_capacityY;
        while (capacityX < requestedCapacityX)
            capacityX *= 2;
        while (capacityY < requestedCapacityY)
            capacityY *= 2;

        if (capacityX > m_capacityX || capacityY > m_capacityY) {
            auto newData = allocate(capacityX, capacityY);
            auto oldData = m_data;

            for (size_type j = 0; j < capacityY; ++j)
                for (size_type i = 0; i < capacityX; ++i) {
                    if (i >= m_sizeX || j >= m_sizeY)
                        ::new (newData + (i + j * capacityX)) value_type();
                    else {
                        auto newVal =
                            ::new (newData + (i + j * capacityX + i)) value_type();
                        auto oldVal = &m_data[i + m_capacityX * j];
                        *newVal = std::move(*oldVal);
                    }
                }

            cleanup(m_data, m_capacityX, m_capacityY);
            m_data = newData;
            m_capacityX = capacityX;
            m_capacityY = capacityY;
        }
    }

 private:
    void check_range(size_type ix) const {
        if (ix >= m_sizeX) NVFLOW_INDEX_OUT_OF_RANGE_ERROR();
    }

    value_type *allocate(size_type capacityX, size_type capacityY) {
        if (capacityX > cached_size_x || capacityY > cached_size_y)
            return (value_type *)Allocable::allocate(sizeof(value_type) * capacityX *
                                                     capacityY);
        else
            return (value_type *)m_cache;
    }

    void cleanup(T *data, size_type capacityX, size_type capacityY) {
        size_type capacity = capacityX * capacityY;
        for (size_t i = 0; i < capacity; ++i)
            data[i].~value_type();
        if (data != (T *)m_cache) Allocable::deallocate(data);
    }

    value_type *m_data;
    size_type m_capacityX;
    size_type m_capacityY;
    size_type m_sizeX;
    size_type m_sizeY;
    alignas(alignof(T)) uint8_t m_cache[sizeof(value_type) * cached_size];
};

};  // namespace NvFlow

namespace std {

template <typename T, uint32_t N>
T *begin(NvFlow::VectorCached<T, N> &con) {
    return con.data();
}

template <typename T, uint32_t N>
const T *begin(const NvFlow::VectorCached<T, N> &con) {
    return con.data();
}

template <typename T, uint32_t N>
T *end(NvFlow::VectorCached<T, N> &con) {
    return con.data() + con.size();
}

template <typename T, uint32_t N>
const T *end(const NvFlow::VectorCached<T, N> &con) {
    return con.data() + con.size();
}

};  // namespace std

#endif /* VECTORCACHED_H */
