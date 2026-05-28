#pragma once

#include "bindings/common.h"

#include <unordered_map>

#include "rxmesh/attribute.h"
#include "rxmesh/reduce_handle.h"

namespace pyrxmesh_py {

template <typename T, typename HandleT>
using AttrPtr = std::shared_ptr<rxmesh::Attribute<T, HandleT>>;

template <typename T>
inline constexpr bool is_attribute_matrix_copy_type_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, int32_t>;

template <typename T, typename HandleT>
void ensure_host_readable(rxmesh::Attribute<T, HandleT>& attr)
{
    if (attr.is_host_allocated()) {
        return;
    }
    if (!attr.is_device_allocated()) {
        throw std::runtime_error(
            "Attribute is not allocated on HOST or DEVICE, so it cannot be "
            "copied to Python.");
    }
    attr.move(rxmesh::DEVICE, rxmesh::HOST);
}

template <typename T, typename HandleT>
void ensure_host_writable(rxmesh::Attribute<T, HandleT>& attr)
{
    if (attr.is_host_allocated()) {
        return;
    }
    if (!attr.is_device_allocated()) {
        throw std::runtime_error(
            "Attribute is not allocated on HOST or DEVICE, so it cannot be "
            "populated from Python.");
    }
    attr.move(rxmesh::DEVICE, rxmesh::HOST);
}

template <typename T, typename HandleT>
void ensure_allocated(rxmesh::Attribute<T, HandleT>& attr,
                      rxmesh::locationT              location)
{
    if (((location & rxmesh::HOST) == rxmesh::HOST) &&
        !attr.is_host_allocated()) {
        if (!attr.is_device_allocated()) {
            throw std::runtime_error(
                "Attribute has no allocation to copy from while ensuring "
                "HOST.");
        }
        attr.move(rxmesh::DEVICE, rxmesh::HOST);
    }

    if (((location & rxmesh::DEVICE) == rxmesh::DEVICE) &&
        !attr.is_device_allocated()) {
        if (!attr.is_host_allocated()) {
            throw std::runtime_error(
                "Attribute has no allocation to copy from while ensuring "
                "DEVICE.");
        }
        attr.move(rxmesh::HOST, rxmesh::DEVICE);
    }
}

inline uint32_t reduction_column(std::optional<uint32_t> column)
{
    using namespace rxmesh;
    return column.value_or(INVALID32);
}

template <typename T>
void ensure_reduction_dtype()
{
    if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_same_v<T, int32_t>)) {
        throw std::invalid_argument(
            "Reductions are support only for float32, float64, or int32.");
    }
}

struct PyAttributeBase
{
    explicit PyAttributeBase(std::shared_ptr<rxmesh::RXMeshStatic> owner)
        : mesh_owner(std::move(owner))
    {
    }

    virtual ~PyAttributeBase() = default;

    std::shared_ptr<rxmesh::RXMeshStatic> mesh_owner;

    virtual const char* name() const                               = 0;
    virtual DType       dtype() const                              = 0;
    virtual ElementKind element_kind() const                       = 0;
    virtual uint32_t    dim() const                                = 0;
    virtual int         allocated() const                          = 0;
    virtual int         layout() const                             = 0;
    virtual size_t      size() const                               = 0;
    virtual uint32_t    element_count() const                      = 0;
    virtual size_t      bytes() const                              = 0;
    virtual bool        is_host_allocated() const                  = 0;
    virtual bool        is_device_allocated() const                = 0;
    virtual bool        is_tensor_layout() const                   = 0;
    virtual void*       data_ptr(rxmesh::locationT location) const = 0;
    virtual rxmesh::AttributeBase* raw_attribute_base() const      = 0;

    py::tuple shape() const
    {
        return py::make_tuple(element_count(), dim());
    }

    virtual void reset(py::object value, int location)                     = 0;
    virtual void move(int source, int target)                              = 0;
    virtual void copy_from(PyAttributeBase& other, int source, int target) = 0;
    virtual py::array   to_numpy(int location, py::object owner)           = 0;
    virtual py::array   to_numpy_copy(int source)                          = 0;
    virtual void        from_numpy_copy(py::array values, int target)      = 0;
    virtual py::object  to_matrix_copy()                                   = 0;
    virtual void        from_matrix_copy(py::object matrix, int target)    = 0;
    virtual py::object  reduce_sum(std::optional<uint32_t> column)         = 0;
    virtual py::object  reduce_min(std::optional<uint32_t> column)         = 0;
    virtual py::object  reduce_max(std::optional<uint32_t> column)         = 0;
    virtual py::object  norm2(std::optional<uint32_t> column)              = 0;
    virtual py::object  dot(PyAttributeBase&        other,
                            std::optional<uint32_t> column)                = 0;
    virtual py::tuple   argmax(std::optional<uint32_t> column)             = 0;
    virtual py::tuple   argmin(std::optional<uint32_t> column)             = 0;
    virtual py::capsule capsule()                                          = 0;
    virtual py::object  add_like(
         const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
         const std::string&                           name) const                  = 0;
    virtual void export_obj(rxmesh::RXMeshStatic& mesh,
                            const std::string&    filename) = 0;
};

template <typename T, typename HandleT>
struct PyAttribute final : PyAttributeBase
{
    AttrPtr<T, HandleT> attr;

    std::unique_ptr<rxmesh::ReduceHandle<T, HandleT>> m_reducer;

    PyAttribute(std::shared_ptr<rxmesh::RXMeshStatic> owner,
                AttrPtr<T, HandleT>                   attr_ptr)
        : PyAttributeBase(std::move(owner)),
          attr(std::move(attr_ptr))
    {
    }

    PyAttribute(const PyAttribute&)            = delete;
    PyAttribute& operator=(const PyAttribute&) = delete;

    const char* name() const override
    {
        return attr->get_name();
    }

    DType dtype() const override
    {
        return dtype_for<T>();
    }

    ElementKind element_kind() const override
    {
        return element_kind_for<HandleT>();
    }

    uint32_t dim() const override
    {
        return attr->get_num_attributes();
    }

    int allocated() const override
    {
        return static_cast<int>(attr->get_allocated());
    }

    int layout() const override
    {
        return static_cast<int>(attr->get_layout());
    }

    size_t size() const override
    {
        return attr->size();
    }

    uint32_t element_count() const override
    {
        return attr->rows();
    }

    size_t bytes() const override
    {
        return attr->get_total_bytes();
    }

    bool is_host_allocated() const override
    {
        return attr->is_host_allocated();
    }

    bool is_device_allocated() const override
    {
        return attr->is_device_allocated();
    }

    bool is_tensor_layout() const override
    {
        return attr->is_tensor_layout();
    }

    void* data_ptr(rxmesh::locationT location) const override
    {
        return static_cast<void*>(attr->data(location));
    }

    rxmesh::AttributeBase* raw_attribute_base() const override
    {
        return attr.get();
    }

    void reset(py::object value, int location) override
    {
        attr->reset(value.cast<T>(), parse_location(location));
    }

    void move(int source, int target) override
    {
        attr->move(parse_location(source), parse_location(target));
    }


    void copy_from(PyAttributeBase& other, int source, int target) override
    {
        auto* typed_other = dynamic_cast<PyAttribute<T, HandleT>*>(&other);

        const auto src = parse_location(source);
        const auto dst = parse_location(target);
        attr->copy_from(*typed_other->attr, src, dst);
    }

    py::array to_numpy(int location, py::object owner) override
    {
        const auto loc = parse_location(location);
        if (loc != rxmesh::HOST) {
            throw std::invalid_argument(
                "Attribute.to_numpy() only supports Location.HOST. Use "
                "Attribute.to_torch(Location.DEVICE) for CUDA zero-copy "
                "views or Attribute.to_numpy_copy() for copies.");
        }
        if (!attr->is_tensor_layout()) {
            throw std::runtime_error(
                "Attribute.to_numpy() only supports zero-copy views for "
                "Layout.SoA attributes. Use Attribute.to_numpy_copy() for "
                "AoS/AoSoA attributes.");
        }
        if (!attr->is_host_allocated()) {
            throw std::runtime_error(
                "Attribute.to_numpy() requires an existing HOST allocation. "
                "Move or create the attribute on HOST, or use "
                "Attribute.to_numpy_copy().");
        }

        const auto rows = static_cast<py::ssize_t>(attr->rows());
        const auto cols = static_cast<py::ssize_t>(attr->cols());
        return py::array_t<T>({rows, cols},
                              {static_cast<py::ssize_t>(sizeof(T)),
                               static_cast<py::ssize_t>(sizeof(T) * rows)},
                              attr->data(rxmesh::HOST),
                              std::move(owner));
    }

    py::array to_numpy_copy(int source) override
    {
        const auto src = parse_location(source);
        if (src != rxmesh::HOST && src != rxmesh::DEVICE) {
            throw std::invalid_argument(
                "Attribute.to_numpy_copy() source must be Location.HOST or "
                "Location.DEVICE.");
        }
        if (src == rxmesh::DEVICE) {
            if (!attr->is_device_allocated()) {
                throw std::runtime_error(
                    "Attribute.to_numpy_copy() cannot copy from DEVICE "
                    "because DEVICE is not allocated.");
            }
        } else {
            ensure_host_readable(*attr);
        }


        py::array_t<T> out({static_cast<py::ssize_t>(attr->rows()),
                            static_cast<py::ssize_t>(attr->cols())});

        auto view = out.template mutable_unchecked<2>();

        for (uint32_t i = 0; i < attr->rows(); ++i) {
            for (uint32_t j = 0; j < attr->cols(); ++j) {
                view(i, j) = (*attr)(i, j);
            }
        }
        return out;
    }

    void from_numpy_copy(py::array values, int target) override
    {
        const auto dst = parse_location(target);

        validate_numpy_shape(values, "Attribute.from_numpy_copy()");
        ensure_host_writable(*attr);

        py::array_t<T, py::array::c_style | py::array::forcecast> typed(values);
        const py::buffer_info info = typed.request();
        if (info.ndim == 1) {
            auto view = typed.template unchecked<1>();
            for (uint32_t i = 0; i < attr->rows(); ++i) {
                (*attr)(i, 0) = view(i);
            }
        } else {
            auto view = typed.template unchecked<2>();
            for (uint32_t i = 0; i < attr->rows(); ++i) {
                for (uint32_t j = 0; j < attr->cols(); ++j) {
                    (*attr)(i, j) = view(i, j);
                }
            }
        }

        if ((dst & rxmesh::DEVICE) == rxmesh::DEVICE) {
            attr->move(rxmesh::HOST, rxmesh::DEVICE);
        }
    }

    py::object to_matrix_copy() override
    {
        if constexpr (!is_attribute_matrix_copy_type_v<T>) {
            throw std::invalid_argument(
                "Attribute.to_matrix_copy() supports float32, float64, and "
                "int32 attributes.");
        } else {
            ensure_allocated(*attr, rxmesh::HOST);
            auto mat = attr->template to_matrix<Eigen::ColMajor>();
            DenseMatrixVariant matrix = mat;
            return py::cast(std::make_shared<PyDenseMatrix>(
                std::move(matrix), rxmesh::LOCATION_ALL));
        }
    }

    void from_matrix_copy(py::object matrix, int target) override
    {
        if constexpr (!is_attribute_matrix_copy_type_v<T>) {
            throw std::invalid_argument(
                "Attribute.from_matrix_copy() supports float32, float64, and "
                "int32 attributes.");
        } else {
            const auto dst = parse_location(target);
            auto dense = matrix.template cast<std::shared_ptr<PyDenseMatrix>>();

            if (dtype() != parse_dtype(dense->dtype()) ||
                element_count() != static_cast<uint32_t>(dense->rows()) ||
                dim() != static_cast<uint32_t>(dense->cols())) {
                throw std::invalid_argument(
                    "Attribute.from_matrix_copy() requires matching dtype and "
                    "shape.");
            }

            dense->ensure_allocated(rxmesh::HOST);
            ensure_host_writable(*attr);

            std::visit(
                [&](auto& mat) {
                    using MatT = std::decay_t<decltype(mat)>;
                    if constexpr (std::is_same_v<
                                      T,
                                      typename MatT::element_type::Type>) {
                        attr->template from_matrix<Eigen::ColMajor>(mat.get());
                    } else {
                        throw std::invalid_argument(
                            "Attribute.from_matrix_copy() requires an exactly "
                            "matching DenseMatrix dtype.");
                    }
                },
                dense->matrix);

            if ((dst & rxmesh::DEVICE) == rxmesh::DEVICE) {
                attr->move(rxmesh::HOST, rxmesh::DEVICE);
            }
        }
    }

    py::object reduce_sum(std::optional<uint32_t> column) override
    {
        return reduce_with(std::move(column), cub::Sum(), [](auto type_tag) {
            using ValueT = decltype(type_tag);
            return ValueT{};
        });
    }

    py::object reduce_min(std::optional<uint32_t> column) override
    {
        return reduce_with(std::move(column), cub::Min(), [](auto type_tag) {
            using ValueT = decltype(type_tag);
            return std::numeric_limits<ValueT>::max();
        });
    }

    py::object reduce_max(std::optional<uint32_t> column) override
    {
        return reduce_with(std::move(column), cub::Max(), [](auto type_tag) {
            using ValueT = decltype(type_tag);
            return std::numeric_limits<ValueT>::lowest();
        });
    }

    py::object norm2(std::optional<uint32_t> column) override
    {
        ensure_reduction_dtype<T>();
        ensure_allocated(*attr, rxmesh::DEVICE);
        auto& reducer = get_reducer();
        return py::cast(
            reducer.norm2(*attr, reduction_column(std::move(column))));
    }

    py::object dot(PyAttributeBase&        other,
                   std::optional<uint32_t> column) override
    {
        auto* typed_other = dynamic_cast<PyAttribute<T, HandleT>*>(&other);
        if (!typed_other || dim() != other.dim()) {
            throw std::invalid_argument(
                "Attribute.dot() requires exactly matching typed attributes.");
        }
        ensure_reduction_dtype<T>();
        ensure_allocated(*attr, rxmesh::DEVICE);
        ensure_allocated(*typed_other->attr, rxmesh::DEVICE);
        auto& reducer = get_reducer();
        return py::cast(reducer.dot(
            *attr, *typed_other->attr, reduction_column(std::move(column))));
    }

    py::tuple argmax(std::optional<uint32_t> column) override
    {
        return argminmax<true>(std::move(column));
    }

    py::tuple argmin(std::optional<uint32_t> column) override
    {
        return argminmax<false>(std::move(column));
    }

    py::capsule capsule() override
    {
        auto* capsule_data         = new pyrxmesh::AttributeCapsule();
        capsule_data->abi_version  = plugin_abi_version;
        capsule_data->build_config = build_config_tag;
        capsule_data->element_kind = static_cast<pyrxmesh::ElementKind>(
            static_cast<uint32_t>(element_kind()));
        capsule_data->dtype =
            static_cast<pyrxmesh::DType>(static_cast<uint32_t>(dtype()));
        capsule_data->num_attributes = dim();
        capsule_data->attribute      = attr.get();

        return py::capsule(capsule_data,
                           pyrxmesh::attribute_capsule_name,
                           destroy_attribute_capsule);
    }

    py::object add_like(const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
                        const std::string& name) const override
    {
        auto new_attr =
            mesh->template add_attribute_like<T, HandleT>(name, *attr);
        return py::cast(
            std::make_shared<PyAttribute<T, HandleT>>(mesh, new_attr));
    }

    void export_obj(rxmesh::RXMeshStatic& mesh,
                    const std::string&    filename) override
    {
        if constexpr (std::is_same_v<HandleT, rxmesh::VertexHandle> &&
                      (std::is_same_v<T, float> || std::is_same_v<T, double>)) {
            ensure_allocated(*attr, rxmesh::HOST);
            mesh.export_obj<T>(filename, *attr);
        } else {
            throw std::invalid_argument(
                "RXMeshStatic.export_obj() coords must be a float32 or "
                "float64 vertex attribute.");
        }
    }

   private:
    rxmesh::ReduceHandle<T, HandleT>& get_reducer()
    {
        if (!m_reducer) {
            m_reducer =
                std::make_unique<rxmesh::ReduceHandle<T, HandleT>>(*attr);
        }
        return *m_reducer;
    }
    void validate_numpy_shape(const py::array& values, const char* api) const
    {
        const py::buffer_info info = values.request();
        if (info.ndim != 1 && info.ndim != 2) {
            throw std::invalid_argument(std::string(api) +
                                        " expects a 1D or 2D array.");
        }
        if (static_cast<uint32_t>(info.shape[0]) != attr->rows()) {
            throw std::invalid_argument(
                std::string(api) +
                " row count does not match the mesh element count.");
        }
        if (info.ndim == 1 && attr->cols() != 1) {
            throw std::invalid_argument(
                std::string(api) +
                " received a 1D array for a multi-column attribute.");
        }
        if (info.ndim == 2 &&
            static_cast<uint32_t>(info.shape[1]) != attr->cols()) {
            throw std::invalid_argument(
                std::string(api) +
                " column count does not match the attribute dimension.");
        }
    }

    template <typename ReductionOp, typename InitFactory>
    py::object reduce_with(std::optional<uint32_t> column,
                           ReductionOp             reduction_op,
                           InitFactory             init)
    {
        ensure_reduction_dtype<T>();
        ensure_allocated(*attr, rxmesh::DEVICE);
        auto& reducer = get_reducer();
        return py::cast(reducer.reduce(*attr,
                                       reduction_op,
                                       static_cast<T>(init(T{})),
                                       reduction_column(std::move(column))));
    }

    template <bool IsMax>
    py::tuple argminmax(std::optional<uint32_t> column)
    {
        ensure_reduction_dtype<T>();
        ensure_allocated(*attr, rxmesh::DEVICE);
        auto& reducer = get_reducer();
        auto  result  = [&]() {
            if constexpr (IsMax) {
                return reducer.arg_max(*attr, reduction_column(column));
            } else {
                return reducer.arg_min(*attr, reduction_column(column));
            }
        }();
        return py::make_tuple(result.key, result.value);
    }
};

template <typename HandleT, typename T>
py::object make_attribute_object(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
    const std::string&                           name,
    uint32_t                                     dim,
    rxmesh::locationT                            location,
    rxmesh::layoutT                              layout)
{
    auto attr =
        mesh->template add_attribute<T, HandleT>(name, dim, location, layout);
    return py::cast(std::make_shared<PyAttribute<T, HandleT>>(mesh, attr));
}

template <typename HandleT>
py::object add_typed_attribute(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
    const std::string&                           name,
    const std::string&                           dtype,
    uint32_t                                     dim,
    int                                          location,
    int                                          layout)
{
    const DType parsed_dtype = parse_dtype(dtype);
    const auto  loc          = parse_location(location);
    const auto  mem_layout   = parse_layout(layout);

    switch (parsed_dtype) {
        case DType::Float32:
            return make_attribute_object<HandleT, float>(
                mesh, name, dim, loc, mem_layout);
        case DType::Float64:
            return make_attribute_object<HandleT, double>(
                mesh, name, dim, loc, mem_layout);
        case DType::Int32:
            return make_attribute_object<HandleT, int32_t>(
                mesh, name, dim, loc, mem_layout);
        case DType::Int8:
            return make_attribute_object<HandleT, int8_t>(
                mesh, name, dim, loc, mem_layout);
        default:
            throw std::invalid_argument("Unsupported RXMesh attribute dtype.");
    }
}

inline py::object add_attribute_like(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
    const std::string&                           name,
    const PyAttributeBase&                       other)
{
    if (mesh.get() != other.mesh_owner.get()) {
        throw std::invalid_argument(
            "RXMeshStatic.add_attribute_like() requires an attribute from the "
            "same mesh.");
    }

    return other.add_like(mesh, name);
}

}  // namespace pyrxmesh_py
