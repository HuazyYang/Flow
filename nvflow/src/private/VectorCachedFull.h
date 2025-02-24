#ifndef VECTORCACHE_H
#define VECTORCACHE_H
#include <vector>
#include <cstdint>
#include <memory>
#include "Allocable.h"
#include <stdexcept>

namespace NvFlow {

namespace details {
template <typename T>
typename std::enable_if<std::is_destructible<T>::value, void>::type destruct(T* ptr) {
    ptr->~T();
}

template <typename T>
typename std::enable_if<!std::is_destructible<T>::value, void>::type destruct(T* ptr) {}

template <typename Allocator, typename T,
          std::enable_if_t<std::is_destructible_v<T>, int> = 0>
void destroy_range(T* start, T* end, Allocator& al) {
    while (start != end) {
        al.destroy(start++);
    }
}

template <typename Allocator, typename T,
          std::enable_if_t<!std::is_destructible_v<T>, int> = 0>
void destroy_range(T* start, T* end, Allocator& al) {}

struct ValueInitTag {
    explicit ValueInitTag() = default;
};

template <typename Allocator, typename T>
T* construct_fill_n(T* start, T* end, Allocator& al, const T& val) {
    while (start != end) {
        al.construct(start++, val);
    }
    return end;
}
template <typename Allocator, typename T>
T* construct_fill_n(T* start, T* end, Allocator& al, const ValueInitTag& tag) {
    while (start != end) {
        al.construct(start++);
    }
    return end;
}

template <typename Allocator, typename T, typename Iter>
T* copy_construct_range(Iter start, Iter end, T* dest, const Allocator& al) {
    for (; start != end; ++start) {
        al.construct(dest++, *start);
    }
    return dest;
}

template <typename Allocator, typename T, typename Iter>
T* move_construct_range(Iter start, Iter end, T* dest, const Allocator& al) {
    for (; start != end; ++start) {
        al.construct(dest++, std::move(*start));
    }
    return dest;
}

template <typename T>
T* move_backward(T* start, T* end, T* dest) {
    while (start != end)
        *--dest = std::move(*--end);
    return dest;
}

template <typename Alloc>
struct AllocTemporary {
    template <typename... Args>
    AllocTemporary(Args&&... args) {
        _pobj = _alloc.allocate(1);
        _alloc.construct(_pobj, std::forward<Args>(args)...);
    }

    AllocTemporary() {
        _alloc.destroy(_pobj);
        _alloc.deallocate(_pobj);
    }

    typename Alloc::value_type& get_val() { return _pobj; }

    const typename Alloc::value_type& get_val() const { return _pobj; }

    Alloc _alloc;
    typename Alloc::value_type* _pobj;
};

template <typename T, size_t N>
struct CachedAllocator {
    using value_type = T;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // template <class Other>
    // struct rebind {
    //     using other = CachedAllocator<Other, N>;
    // };

    constexpr CachedAllocator() noexcept {}
    // template <class Other>
    // constexpr CachedAllocator(const CachedAllocator<Other, N>&) noexcept {
    // }
    constexpr CachedAllocator(const CachedAllocator&) noexcept = default;
    ~CachedAllocator() = default;

    constexpr CachedAllocator& operator=(CachedAllocator&) = default;

    bool operator==(const CachedAllocator& other) noexcept { return this == &other; }

    bool operator!=(const CachedAllocator& other) noexcept { return this != &other; }

    T* allocate(std::size_t count) {
        if (count <= N)
            return (T*)_cache_mem;
        else
            return (T*)Allocable::allocate(count * sizeof(T));
    }

    pointer address(reference r) { return std::addressof(r); }
    const_pointer address(const_reference r) { return std::addressof(r); }

    void deallocate(T* p, std::size_t count) {
        if (count < N)
            return;
        else
            return Allocable::deallocate(p);
    }

    inline size_type max_size() const {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    //    construction/destruction
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (p) U(std::forward<Args>(args)...);
    }

    inline void destroy(pointer p) { destruct(p); }

 private:
    alignas(alignof(T)) uint8_t _cache_mem[sizeof(T) * N];
};

template <typename T, size_t N>
class VectorCached {
 public:
    using value_type = T;
    using allocator_type = CachedAllocator<T, N>;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    using reference = typename allocator_type::reference;
    using const_reference = typename allocator_type::const_pointer;
    using size_type = typename allocator_type::size_type;
    using difference_type = typename allocator_type::difference_type;

    using iterator = pointer;
    using const_iterator = const_pointer;

    VectorCached() : _first(nullptr), _last(nullptr), _end(nullptr) {}

    VectorCached(size_type new_size) : VectorCached{} { resize(new_size); }

    VectorCached(size_type new_size, const T& val) : VectorCached{} {
        resize(new_size, val);
    }

    template <typename Iter>
    VectorCached(Iter first, Iter last) : VectorCached{} {
        const size_type new_size = std::distance(first, last);
        if (new_size) {
            _allocate_exactly(new_size);
            _last = _first + new_size;
            copy_construct_range(first, last, _first, _alloc);
        }
    }

    VectorCached(std::initializer_list<T> lst) : VectorCached{} {
        const size_type new_size = lst.size();
        if (new_size) {
            _allocate_exactly(new_size);
            _last = _first + new_size;
            copy_construct_range(lst.begin(), lst.end(), _first, _alloc);
        }
    }

    VectorCached(const VectorCached& right) : VectorCached{} {
        const size_type new_size = right.size();
        if (new_size) {
            _allocate_exactly(new_size);
            _last = _first + new_size;
            copy_construct_range(right._first, right._last, _first, _alloc);
        }
    }

    VectorCached(VectorCached&& right) noexcept : VectorCached{} { this->swap(right); }

    VectorCached& operator=(const VectorCached& right) {
        const size_type new_size = right.size();
        if (new_size && this != std::addressof(right)) {
            _destroy_reallocate_exactly(new_size);
            _last = _first + new_size;
            copy_construct_range(right._first, right._last, _first, _alloc);
        }
        return *this;
    }

    VectorCached& operator=(VectorCached&& right) noexcept {
        this->swap(right);
        return *this;
    }

    ~VectorCached() { _tidy(); }

    size_type max_size() const { return _alloc.max_size(); }

    size_type capacity() const { return _end - _first; }

    size_type size() const { return _last - _first; }

    bool empty() const { return _first != _last; }

    value_type* data() { return _first; }

    const value_type* data() const { return _first; }

    void resize(size_type new_size) { _resize(new_size, ValueInitTag{}); }

    void resize(size_type new_size, const value_type& val) { _resize(new_size, val); }

    void reserve(size_type new_capacity) {
        if (new_capacity > capacity()) {
            _reallocate_exactly(new_capacity);
        }
    }

    void clear() {
        destroy_range(_first, _last, _alloc);
        _last = _first;
    }

    void swap(VectorCached& right) noexcept { _swap(right); }

    void shrink_to_fit() {
        if (_last != _end) {
            _reallocate_exactly(_last - _first);
        }
    }

    void push_back(const T& val) { emplace_back(val); }

    void push_back(T&& val) { emplace_back(std::move(val)); }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        const size_type old_capacity = capacity();
        const size_type old_size = size();
        const size_type new_size = old_size + 1;
        pointer old_last = _last;

        if (old_size == max_size()) _xlength();

        if (old_capacity < new_size) {
            const size_type new_capacity = _calculate_growth(new_size);
            _reallocate_exactly(new_capacity);
        }

        _alloc.construct(_last, std::forward<Args>(args)...);
        _last += 1;

        return *old_last;
    }

    template <typename... Args>
    reference emplace(const_iterator where, Args&&... args) {
        auto old_last = _last;
        if (old_last != _end) {
            if (where == old_last) {
                _alloc.construct(old_last, std::forward<Args>(args)...);
                _last += 1;
            } else {
                // Handle alias
                AllocTemporary<CachedAllocator<value_type, 1>> obj(
                    std::forward<Args>(args)...);

                _alloc.construct(old_last, std::move(old_last[-1]));
                _last += 1;
                move_backward(where, old_last - 1, old_last);
                _alloc.destroy(where);
                _alloc.construct(where, std::move(obj.get_val()));
            }

            return *where;
        }

        return *_emplace_reallocate(where, std::forward<Args>(args)...);
    }

    iterator insert(const_iterator where, const T& val) { return emplace(where, val); }

    iterator insert(const_pointer where, T&& val) { return emplace(where, std::move(val)); }

    iterator insert(const_pointer where, const size_type count, const T& val) {
        const auto whereoff = static_cast<size_type>(where - _first);
        const auto unused_capacity = static_cast<size_type>(_end - _last);
        const bool one_at_back = count == 1 && where == _last;

        if (count == 0) {
            // Noting to do
        } else if (count > unused_capacity) {
            const auto old_size = size();

            if (count > max_size() - old_size) _xlength();

            const auto new_size = old_size + count;
            const auto new_capacity = _calculate_growth(new_size);

            const auto new_first = _alloc.allocate(new_capacity);
            const auto constructed_first = new_first + whereoff;
            const auto constructed_last = constructed_first + count;
            construct_fill_n(constructed_first, constructed_last, _alloc, val);
            move_construct_range(_first, _first + whereoff, new_first, _alloc);
            move_construct_range(_first + whereoff, _last, constructed_last, _alloc);
            destroy_range(_first, _last, _alloc);
            _alloc.deallocate(_first, capacity());

            _first = new_first;
            _end = _first + new_capacity;
            _last = _first + new_size;

            where = _first + whereoff;
        } else if (one_at_back) {
            _alloc.construct(_last, val);
            _last += 1;
        } else {
            const AllocTemporary<CachedAllocator<value_type, 1>> temp(val);
            const auto affected_elements = static_cast<size_type>(_last - where);
            pointer new_last;

            if (count > affected_elements) {
                new_last = construct_fill_n(_last, _last + (count - affected_elements),
                                            _alloc, val);
                new_last = move_construct_range(where, _last, new_last, _alloc);
                std::fill(where, _last, temp);
            } else {
                new_last = move_construct_range(_last - count, _last, _last, _alloc);
                move_backward(where, _last - count, _last);
                std::fill(where, where + count, temp);
            }

            _last = new_last;
        }

        return where;
    }

 private:
    size_type _calculate_growth(size_type new_size) {
        const size_type old_capacity = capacity();
        const size_type _max = max_size();

        if (old_capacity > _max - old_capacity / 4) return _max;

        size_type grow_by = old_capacity / 4;
        grow_by = grow_by < 4 ? 4 : 1024 < grow_by ? 1024 : grow_by;

        if (new_size < old_capacity + grow_by) new_size = old_capacity + grow_by;

        return new_size;
    }

    template <typename T2>
    void _resize(size_type new_size, const T2& init_val) {
        const size_type old_size = size();

        if (new_size < old_size) {
            const pointer new_last = _first + new_last;
            destroy_range(new_last, _last, _alloc);
            _last = new_last;
            return;
        }

        if (old_size < new_size) {
            // Append
            auto old_capacity = capacity();
            if (old_capacity < new_size) {
                auto new_capacity = _calculate_growth(new_size);
                _reallocate_exactly(new_capacity);
            } else
                _last = _first + new_size;

            construct_fill_n(_first + old_size, _last, init_val);
        }
    }

    void _reallocate_exactly(const size_type new_capacity) {
        if (new_capacity > max_size()) {
            _xlength();
        }

        auto new_first = _alloc.allocate(new_capacity);
        const size_type sz = size();

        move_construct_range(_first, _last, new_first, _alloc);
        destroy_range(_first, _last, _alloc);
        _alloc.deallocate(_first, capacity());

        _first = new_first;
        _last = _first + sz;
        _end = _first + new_capacity;
    }

    void _destroy_reallocate_exactly(const size_type new_capacity) {
        if (new_capacity > max_size()) {
            _xlength();
        }

        destroy_range(_first, _last, _alloc);

        if (new_capacity != capacity()) {
            _alloc.deallocate(_first, capacity());
            _first = _alloc.allocate(new_capacity);
            _end = _first + new_capacity;
        }
        _last = _first;
    }

    void _allocate_exactly(const size_type new_capacity) {
        if (new_capacity > max_size()) {
            _xlength();
        }
        _first = _alloc.allocate(new_capacity);
        _end = _first + new_capacity;
        _last = _first;
    }

    template <typename... Args>
    pointer _emplace_reallocate(pointer where, Args&&... args) {
        const size_type old_size = size();
        const size_type whereoff = static_cast<size_type>(where - _first);

        if (old_size == max_size()) _xlength();

        const size_type new_size = old_size + 1;
        const size_type new_capacity = _calculate_growth(new_size);

        const pointer new_first = _alloc.allocate(new_capacity);

        _alloc.construct(new_first + whereoff, std::forward<Args>(args)...);

        if (where == _last) {
            // At back
            move_construct_range(_first, _last, new_first, _alloc);
        } else {
            move_construct_range(_first, where, new_first, _alloc);
            move_construct_range(where, _last, new_first + whereoff + 1, _alloc);
        }
        destroy_range(_first, _last, _alloc);
        _alloc.deallocate(_first, capacity());

        _first = new_first;
        _last = _first + new_size;
        _end = _first + new_capacity;

        return _first + whereoff;
    }

    void _xlength() { throw std::length_error("std::vector too long"); }

    void _swap(VectorCached& right) noexcept {
        if (this != std::addressof(right)) {
            bool cached1 = (void*)_first == (void*)std::addressof(_alloc);
            bool cached2 = (void*)right._first == (void*)std::addressof(right._alloc);
            size_type sz1 = size();
            size_type sz2 = right.size();
            size_type cap1 = capacity();
            size_type cap2 = right.capacity();

            if (cached1 && cached2) {
                std::swap(_alloc, right._alloc);

                _last = _first + sz2;
                _end = _first + cap2;
                right._last = right._first + sz1;
                right._end = right._first + cap1;
            } else if (cached1) {
                right._alloc = _alloc;  // Copy content of this cache

                _first = right._first;
                _last = right._last;
                _end = right._end;
                right._first = reinterpret_cast<value_type*>(std::addressof(right._alloc));
                right._last = right._first + sz1;
                right._end = right._first + cap1;
            } else if (cached2) {
                _alloc = right._alloc;  // Copy content of the right cache.

                right._first = _first;
                right._last = _last;
                right._end = _end;
                _first = reinterpret_cast<value_type*>(std::addressof(_alloc));
                _last = right._first + sz2;
                _end = right._first + cap2;
            } else {
                std::swap(_first, right._first);
                std::swap(_last, right._last);
                std::swap(_end, right._end);
            }
        }
    }

    void _tidy() {
        if (_first != _last) destroy_range(_first, _last, _alloc);
        if (_first != _end) _alloc.deallocate(_first, capacity());
        _first = _end = _last = nullptr;
    }

    allocator_type _alloc;
    value_type* _first;
    value_type* _last;
    value_type* _end;
};

template <typename Ty, size_t N>
bool operator==(const VectorCached<Ty, N>& left, const VectorCached<Ty, N>& right) {
    if (left.size() != right.size()) return false;

    return std::equal(left.data(), left.data() + left.size(), right.data());
}

template <typename Ty, size_t N>
bool operator!=(const VectorCached<Ty, N>& left, const VectorCached<Ty, N>& right) {
    return !(left == right);
}

};  // namespace details

}  // namespace NvFlow

namespace std {
template <typename Ty, size_t N>
void swap(NvFlow::details::VectorCached<Ty, N>& left,
          NvFlow::details::VectorCached<Ty, N>& right) {
    left.swap(right);
}

template <typename Ty, size_t N>
void swap(NvFlow::details::VectorCached<Ty, N>&& left,
          NvFlow::details::VectorCached<Ty, N>&& right) {
    left.swap(right);
}

template <typename Ty, size_t N>
void swap(NvFlow::details::VectorCached<Ty, N>&& left,
          NvFlow::details::VectorCached<Ty, N>& right) {
    left.swap(right);
}

template <typename Ty, size_t N>
void swap(NvFlow::details::VectorCached<Ty, N>& left,
          NvFlow::details::VectorCached<Ty, N>&& right) {
    left.swap(right);
}

};  // namespace std

#endif /* VECTORCACHE_H */
