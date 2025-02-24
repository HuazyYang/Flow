#ifndef NVFLOWUTILS_H
#define NVFLOWUTILS_H
#include <wrl/client.h>
#include "Types.h"
#include "VectorCached.h"

namespace NvFlow {

template <typename T>
using ComPtr = Microsoft::WRL::ComPtr<T>;

#define NVFLOW_IMPLEMENT_OBJECT_REFERENCE() \
    uint32_t addRef() override {            \
        return Object::addRef();            \
    }                                       \
    uint32_t release() override {           \
        return Object::release();           \
    }

template <typename Timpl, typename Titf>
Timpl* implCast(Titf* ptr) {
    return static_cast<Timpl*>(ptr);
}

template <typename Timpl, typename Titf>
Timpl* implSafeCast(Titf* ptr) {
    return dynamic_cast<Timpl*>(ptr);
}

template <typename T, std::enable_if_t<std::is_base_of_v<IUnknown, T>, int> = 0>
void SafeRelease(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

template <typename T, size_t N, std::enable_if_t<std::is_base_of_v<IUnknown, T>, int> = 0>
void SafeRelease(VectorCached<T*, N>& a) {
    for (auto& v : a) {
        if (v) {
            v->Release();
            v = nullptr;
        }
    }
}

// The main advantage of AutoPtr over the std::shared_ptr is that you can
// attach the same raw pointer to different smart pointers.
//
// For instance, the following code will crash since p will be released twice:
//
// auto *p = new char;
// std::shared_ptr<char> pTmp1(p);
// std::shared_ptr<char> pTmp2(p);
// ...

// This code, in contrast, works perfectly fine:
//
// ObjectBase *pRawPtr(new ObjectBase);
// AutoPtr<ObjectBase> pSmartPtr1(pRawPtr);
// AutoPtr<ObjectBase> pSmartPtr2(pRawPtr);
// ...

// Other advantage is that weak pointers remain valid until the
// object is alive, even if all smart pointers were destroyed:
//
// WeakPtr<ObjectBase> pWeakPtr(pSmartPtr1);
// pSmartPtr1.Release();
// auto pSmartPtr3 = pWeakPtr.Lock();
// ..

// Weak pointers can also be attached directly to the object:
// WeakPtr<ObjectBase> pWeakPtr(pRawPtr);
//

namespace details {

// T should be the AutoPtr<T> or a derived type of it, not just the interface
template <typename T>
class AutoPtrRef {
    using InterfaceType = typename T::InterfaceType;

 public:
    AutoPtrRef(T* ptr) throw() { this->ptr_ = ptr; }

    // Conversion operators
    operator void**() const throw() {
        return reinterpret_cast<void**>(this->ptr_->releaseAndGetAddressOf());
    }

    // This is our operator AutoPtr<U> (or the latest derived class from AutoPtr (e.g.
    // WeakRef))
    operator T*() throw() {
        *this->ptr_ = nullptr;
        return this->ptr_;
    }

    // We define operator InterfaceType**() here instead of on ComPtrRefBase<T>, since
    // if InterfaceType is IObject or IInspectable, having it on the base will collide.
    operator InterfaceType**() throw() { return this->ptr_->releaseAndGetAddressOf(); }

    // This is used for WIID_PPV_ARGS in order to do __uuid_of(**(ppType)).
    // It does not need to clear  ptr_ at this point, it is done at
    // WIID_PPV_ARGS_Helper(AutoPtrRef&) later in this file.
    InterfaceType* operator*() throw() { return this->ptr_->get(); }

    // Explicit functions
    InterfaceType* const* getAddressOf() const throw() {
        return this->ptr_->getAddressOf();
    }

    InterfaceType** releaseAndGetAddressOf() throw() {
        return this->ptr_->releaseAndGetAddressOf();
    }

 private:
    T* ptr_;
};

template <bool Test, typename T = void>
using EnableIf = std::enable_if<Test, T>;

template <typename Tp, typename Ty>
using IsConvertible = std::is_convertible<Tp, Ty>;

template <typename Base, typename Derived>
using IsBaseOf = std::is_base_of<Base, Derived>;

template <typename Tp, typename Ty>
using IsSame = std::is_same<Tp, Ty>;

using nullptr_t = std::nullptr_t;

using BoolType = bool;

}  // namespace details

template <typename T>
class AutoPtr {
 public:
    typedef T InterfaceType;

 protected:
    InterfaceType* ptr_;
    template <class U>
    friend class AutoPtr;

    void internalAddRef() const throw() {
        if (ptr_ != nullptr) {
            ptr_->addRef();
        }
    }

    uint32_t InternalRelease() throw() {
        uint32_t ref = 0;
        T* temp = ptr_;

        if (temp != nullptr) {
            ptr_ = nullptr;
            ref = temp->release();
        }

        return ref;
    }

 public:
#pragma region constructors
    AutoPtr() throw() : ptr_(nullptr) {}

    AutoPtr(details::nullptr_t) throw() : ptr_(nullptr) {}

    template <class U>
    AutoPtr(U* other) throw() : ptr_(other) {
        internalAddRef();
    }

    AutoPtr(const AutoPtr& other) throw() : ptr_(other.ptr_) { internalAddRef(); }

    // copy constructor that allows to instantiate class when U* is convertible to T*
    template <class U>
    AutoPtr(const AutoPtr<U>& other,
            typename details::EnableIf<details::IsConvertible<U*, T*>::value,
                                       void*>::type* = 0) throw()
        : ptr_(other.ptr_) {
        internalAddRef();
    }

    AutoPtr(AutoPtr&& other) throw() : ptr_(nullptr) {
        if (this != reinterpret_cast<AutoPtr*>(&reinterpret_cast<unsigned char&>(other))) {
            swap(other);
        }
    }

    // Move constructor that allows instantiation of a class when U* is convertible to T*
    template <class U>
    AutoPtr(AutoPtr<U>&& other,
            typename details::EnableIf<details::IsConvertible<U*, T*>::value,
                                       void*>::type* = 0) throw()
        : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }
#pragma endregion

#pragma region destructor
    ~AutoPtr() throw() { InternalRelease(); }
#pragma endregion

#pragma region assignment
    AutoPtr& operator=(details::nullptr_t) throw() {
        InternalRelease();
        return *this;
    }

    AutoPtr& operator=(T* other) throw() {
        if (ptr_ != other) {
            AutoPtr(other).swap(*this);
        }
        return *this;
    }

    template <typename U>
    AutoPtr& operator=(U* other) throw() {
        AutoPtr(other).swap(*this);
        return *this;
    }

    AutoPtr& operator=(const AutoPtr& other) throw() {
        if (ptr_ != other.ptr_) {
            AutoPtr(other).swap(*this);
        }
        return *this;
    }

    template <class U>
    AutoPtr& operator=(const AutoPtr<U>& other) throw() {
        AutoPtr(other).swap(*this);
        return *this;
    }

    AutoPtr& operator=(AutoPtr&& other) throw() {
        AutoPtr(static_cast<AutoPtr&&>(other)).swap(*this);
        return *this;
    }

    template <class U>
    AutoPtr& operator=(AutoPtr<U>&& other) throw() {
        AutoPtr(static_cast<AutoPtr<U>&&>(other)).swap(*this);
        return *this;
    }
#pragma endregion

#pragma region modifiers
    void swap(AutoPtr&& r) throw() {
        T* tmp = ptr_;
        ptr_ = r.ptr_;
        r.ptr_ = tmp;
    }

    void swap(AutoPtr& r) throw() {
        T* tmp = ptr_;
        ptr_ = r.ptr_;
        r.ptr_ = tmp;
    }
#pragma endregion

    explicit operator details::BoolType() const noexcept { return ptr_ != nullptr; }

    T* get() const throw() { return ptr_; }

    operator T*() const noexcept { return ptr_; }

    InterfaceType* operator->() const throw() { return ptr_; }

    details::AutoPtrRef<AutoPtr<T>> operator&() throw() {
        return details::AutoPtrRef<AutoPtr<T>>(this);
    }

    const details::AutoPtrRef<const AutoPtr<T>> operator&() const throw() {
        return details::AutoPtrRef<const AutoPtr<T>>(this);
    }

    T* const* getAddressOf() const throw() { return &ptr_; }

    T** getAddressOf() throw() { return &ptr_; }

    T** releaseAndGetAddressOf() throw() {
        InternalRelease();
        return &ptr_;
    }

    T* detach() throw() {
        T* ptr = ptr_;
        ptr_ = nullptr;
        return ptr;
    }

    void attach(InterfaceType* other) throw() {
        if (ptr_ != nullptr) {
            auto ref = ptr_->release();
            (void)ref;
            // Attaching to the same object only works if duplicate references are being
            // coalesced. Otherwise re-attaching will cause the pointer to be released and
            // may cause a crash on a subsequent dereference.
            ETHEREAL_ASSERT(ref != 0 || ptr_ != other);
        }

        ptr_ = other;
    }

    unsigned long reset() { return InternalRelease(); }
};  // AutoPtr

template <typename T>
AutoPtr<T> TakeOver(T* p) noexcept {
    AutoPtr<T> ret;
    *ret.getAddressOf() = p;
    return ret;
}

// Comparison operators - don't compare COM object identity
template <class T, class U>
bool operator==(const AutoPtr<T>& a, const AutoPtr<U>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.get()() == b.get()();
}

template <class T>
bool operator==(const AutoPtr<T>& a, details::nullptr_t) throw() {
    return a.get()() == nullptr;
}

template <class T>
bool operator==(details::nullptr_t, const AutoPtr<T>& a) throw() {
    return a.get()() == nullptr;
}

template <class T, class U>
bool operator!=(const AutoPtr<T>& a, const AutoPtr<U>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.get()() != b.get()();
}

template <class T>
bool operator!=(const AutoPtr<T>& a, details::nullptr_t) throw() {
    return a.get()() != nullptr;
}

template <class T>
bool operator!=(details::nullptr_t, const AutoPtr<T>& a) throw() {
    return a.get()() != nullptr;
}

template <class T, class U>
bool operator<(const AutoPtr<T>& a, const AutoPtr<U>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.get()() < b.get()();
}

//// details::AutoPtrRef comparisons
template <class T, class U>
bool operator==(const details::AutoPtrRef<AutoPtr<T>>& a,
                const details::AutoPtrRef<AutoPtr<U>>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.getAddressOf() == b.getAddressOf();
}

template <class T>
bool operator==(const details::AutoPtrRef<AutoPtr<T>>& a, details::nullptr_t) throw() {
    return a.getAddressOf() == nullptr;
}

template <class T>
bool operator==(details::nullptr_t, const details::AutoPtrRef<AutoPtr<T>>& a) throw() {
    return a.getAddressOf() == nullptr;
}

template <class T>
bool operator==(const details::AutoPtrRef<AutoPtr<T>>& a, void* b) throw() {
    return a.getAddressOf() == b;
}

template <class T>
bool operator==(void* b, const details::AutoPtrRef<AutoPtr<T>>& a) throw() {
    return a.getAddressOf() == b;
}

template <class T, class U>
bool operator!=(const details::AutoPtrRef<AutoPtr<T>>& a,
                const details::AutoPtrRef<AutoPtr<U>>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.getAddressOf() != b.getAddressOf();
}

template <class T>
bool operator!=(const details::AutoPtrRef<AutoPtr<T>>& a, details::nullptr_t) throw() {
    return a.getAddressOf() != nullptr;
}

template <class T>
bool operator!=(details::nullptr_t, const details::AutoPtrRef<AutoPtr<T>>& a) throw() {
    return a.getAddressOf() != nullptr;
}

template <class T>
bool operator!=(const details::AutoPtrRef<AutoPtr<T>>& a, void* b) throw() {
    return a.getAddressOf() != b;
}

template <class T>
bool operator!=(void* b, const details::AutoPtrRef<AutoPtr<T>>& a) throw() {
    return a.getAddressOf() != b;
}

template <class T, class U>
bool operator<(const details::AutoPtrRef<AutoPtr<T>>& a,
               const details::AutoPtrRef<AutoPtr<U>>& b) throw() {
    static_assert(details::IsBaseOf<T, U>::value || details::IsBaseOf<U, T>::value,
                  "'T' and 'U' pointers must be comparable");
    return a.getAddressOf() < b.getAddressOf();
}

}  // namespace NvFlow

#endif /* NVFLOWUTILS_H */
