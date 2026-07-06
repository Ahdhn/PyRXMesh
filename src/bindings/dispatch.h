#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "pyrxmesh/plugin_api.h"
#include "rxmesh/handle.h"

namespace pyrxmesh_py {

enum class ElementKind : uint32_t
{
    Vertex = static_cast<uint32_t>(pyrxmesh::ElementKind::Vertex),
    Edge   = static_cast<uint32_t>(pyrxmesh::ElementKind::Edge),
    Face   = static_cast<uint32_t>(pyrxmesh::ElementKind::Face),
};

enum class DType : uint32_t
{
    Float32 = static_cast<uint32_t>(pyrxmesh::DType::Float32),
    Float64 = static_cast<uint32_t>(pyrxmesh::DType::Float64),
    Int8    = static_cast<uint32_t>(pyrxmesh::DType::Int8),
    UInt8   = static_cast<uint32_t>(pyrxmesh::DType::UInt8),
    Int16   = static_cast<uint32_t>(pyrxmesh::DType::Int16),
    UInt16  = static_cast<uint32_t>(pyrxmesh::DType::UInt16),
    Int32   = static_cast<uint32_t>(pyrxmesh::DType::Int32),
    UInt32  = static_cast<uint32_t>(pyrxmesh::DType::UInt32),
    Int64   = static_cast<uint32_t>(pyrxmesh::DType::Int64),
    UInt64  = static_cast<uint32_t>(pyrxmesh::DType::UInt64),

};


template <typename>
struct always_false : std::false_type
{
};


template <typename T>
DType dtype_for()
{
    if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DType::Int8;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        return DType::Int16;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        return DType::UInt16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return DType::UInt32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DType::Int64;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return DType::UInt64;
    } else {
        static_assert(always_false<T>::value, "Unsupported PyRXMesh dtype");
    }
}

template <typename T>
const char* dense_dtype_name()
{
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, double>) {
        return "float64";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else {
        static_assert(always_false<T>::value, "Unsupported DenseMatrix dtype");
    }
}

template <typename HandleT>
ElementKind element_kind_for()
{
    using namespace rxmesh;

    if constexpr (std::is_same_v<HandleT, VertexHandle>) {
        return ElementKind::Vertex;
    } else if constexpr (std::is_same_v<HandleT, EdgeHandle>) {
        return ElementKind::Edge;
    } else if constexpr (std::is_same_v<HandleT, FaceHandle>) {
        return ElementKind::Face;
    } else {
        static_assert(always_false<HandleT>::value,
                      "Unsupported PyRXMesh element kind");
    }
}

inline DType parse_dtype(const std::string& dtype)
{
    if (dtype == "float32" || dtype == "float") {
        return DType::Float32;
    }
    if (dtype == "float64" || dtype == "double") {
        return DType::Float64;
    }
    if (dtype == "int32" || dtype == "int") {
        return DType::Int32;
    }
    if (dtype == "int8") {
        return DType::Int8;
    }
    throw std::invalid_argument("Unsupported RXMesh attribute dtype: " + dtype);
}

inline const char* dtype_name(DType dtype)
{
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        case DType::Int32:
            return "int32";
        case DType::Int8:
            return "int8";
        default:
            return "unknown";
    }
}

inline const char* element_kind_name(ElementKind element_kind)
{
    switch (element_kind) {
        case ElementKind::Vertex:
            return "vertex";
        case ElementKind::Edge:
            return "edge";
        case ElementKind::Face:
            return "face";
        default:
            return "unknown";
    }
}


template <typename T>
struct dtype_tag
{
    using type = T;
};

namespace detail {

template <typename Fn, typename T>
using invoke_dispatch_t = std::invoke_result_t<Fn, dtype_tag<T>>;

inline std::string dispatch_error_message(const char* operation, DType dtype)
{
    std::string msg = operation ? operation : "dispatch_dtype";
    msg += ": unsupported dtype '";
    msg += dtype_name(dtype);
    msg += "'.";
    return msg;
}

}  // namespace detail


/**
 * Dispatch on a runtime DType to a generic callable that accepts a
 * dtype_tag<T>. The callable is invoked exactly once with the appropriate
 * type tag. Supports float/double/int32/int8.
 *
 * Usage:
 *   dispatch_dtype(parsed_dtype, [&](auto tag) {
 *       using T = typename decltype(tag)::type;
 *       // ... use T ...
 *   });
 */
template <typename Fn>
auto dispatch_dtype(DType d, Fn&& fn) -> detail::invoke_dispatch_t<Fn, float>
{
    if (d == DType::Float32) {
        return std::forward<Fn>(fn)(dtype_tag<float>{});
    }
    if (d == DType::Float64) {
        return std::forward<Fn>(fn)(dtype_tag<double>{});
    }
    if (d == DType::Int32) {
        return std::forward<Fn>(fn)(dtype_tag<int32_t>{});
    }
    if (d == DType::Int8) {
        return std::forward<Fn>(fn)(dtype_tag<int8_t>{});
    }
    throw std::invalid_argument(
        detail::dispatch_error_message("dispatch_dtype", d));
}


/**
 * Dispatch over numeric dtypes only (float/double/int32). Suitable for
 * DenseMatrix and SparseMatrix factories that do not support int8.
 */
template <typename Fn>
auto dispatch_numeric_dtype(DType d, Fn&& fn)
    -> detail::invoke_dispatch_t<Fn, float>
{
    if (d == DType::Float32) {
        return std::forward<Fn>(fn)(dtype_tag<float>{});
    }
    if (d == DType::Float64) {
        return std::forward<Fn>(fn)(dtype_tag<double>{});
    }
    if (d == DType::Int32) {
        return std::forward<Fn>(fn)(dtype_tag<int32_t>{});
    }
    throw std::invalid_argument(
        detail::dispatch_error_message("dispatch_numeric_dtype", d));
}


/**
 * Dispatch over float/double only. Suitable for solvers and floating-point
 * reductions.
 */
template <typename Fn>
auto dispatch_float_dtype(DType d, Fn&& fn)
    -> detail::invoke_dispatch_t<Fn, float>
{
    if (d == DType::Float32) {
        return std::forward<Fn>(fn)(dtype_tag<float>{});
    }
    if (d == DType::Float64) {
        return std::forward<Fn>(fn)(dtype_tag<double>{});
    }
    throw std::invalid_argument(
        detail::dispatch_error_message("dispatch_float_dtype", d));
}


template <typename Fn>
auto dispatch_dtype_str(const std::string& dtype, Fn&& fn)
    -> detail::invoke_dispatch_t<Fn, float>
{
    return dispatch_dtype(parse_dtype(dtype), std::forward<Fn>(fn));
}


template <typename Fn>
auto dispatch_numeric_dtype_str(const std::string& dtype, Fn&& fn)
    -> detail::invoke_dispatch_t<Fn, float>
{
    return dispatch_numeric_dtype(parse_dtype(dtype), std::forward<Fn>(fn));
}


template <typename Fn>
auto dispatch_float_dtype_str(const std::string& dtype, Fn&& fn)
    -> detail::invoke_dispatch_t<Fn, float>
{
    return dispatch_float_dtype(parse_dtype(dtype), std::forward<Fn>(fn));
}


template <int V>
struct int_tag
{
    static constexpr int value = V;
};


/**
 * Dispatch a runtime int value to one of a fixed set of compile-time int
 * tags. Used for constructing types parameterized by an integer (e.g.
 * HessianSparseMatrix<T, K> where K must be one of {1,2,3,4,6}).
 *
 * Usage:
 *   dispatch_int<1, 2, 3, 4, 6>(k, [&](auto tag) {
 *       constexpr int K = decltype(tag)::value;
 *       // ... use K ...
 *   });
 */
template <int... Vs, typename Fn>
auto dispatch_int(int value, Fn&& fn) -> std::invoke_result_t<Fn, int_tag<0>>
{
    using R = std::invoke_result_t<Fn, int_tag<0>>;

    bool matched = false;
    if constexpr (std::is_void_v<R>) {
        ((value == Vs ? (std::forward<Fn>(fn)(int_tag<Vs>{}), matched = true) :
                        false),
         ...);
        if (!matched) {
            throw std::invalid_argument("dispatch_int: value " +
                                        std::to_string(value) +
                                        " is not in the supported set.");
        }
        return;
    } else {
        R result{};
        ((value == Vs ?
              (result = std::forward<Fn>(fn)(int_tag<Vs>{}), matched = true) :
              false),
         ...);
        if (!matched) {
            throw std::invalid_argument("dispatch_int: value " +
                                        std::to_string(value) +
                                        " is not in the supported set.");
        }
        return result;
    }
}

}  // namespace pyrxmesh_py
