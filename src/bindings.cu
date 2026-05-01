#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"
#include "rxmesh/rxmesh_static.h"
#include "rxmesh/types.h"
#include "rxmesh/util/log.h"

#if USE_POLYSCOPE
#include "polyscope/polyscope.h"
#endif

namespace py = pybind11;

namespace {

constexpr uint32_t plugin_abi_version = PYRXMESH_PLUGIN_ABI_VERSION;
constexpr const char* build_config_tag = PYRXMESH_BUILD_CONFIG;

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
    Int32   = static_cast<uint32_t>(pyrxmesh::DType::Int32),
    Int8    = static_cast<uint32_t>(pyrxmesh::DType::Int8),
};

template <typename T, typename HandleT>
using AttrPtr = std::shared_ptr<rxmesh::Attribute<T, HandleT>>;

using AttributeVariant =
    std::variant<AttrPtr<float, rxmesh::VertexHandle>,
                 AttrPtr<double, rxmesh::VertexHandle>,
                 AttrPtr<int32_t, rxmesh::VertexHandle>,
                 AttrPtr<int8_t, rxmesh::VertexHandle>,
                 AttrPtr<float, rxmesh::EdgeHandle>,
                 AttrPtr<double, rxmesh::EdgeHandle>,
                 AttrPtr<int32_t, rxmesh::EdgeHandle>,
                 AttrPtr<int8_t, rxmesh::EdgeHandle>,
                 AttrPtr<float, rxmesh::FaceHandle>,
                 AttrPtr<double, rxmesh::FaceHandle>,
                 AttrPtr<int32_t, rxmesh::FaceHandle>,
                 AttrPtr<int8_t, rxmesh::FaceHandle>>;

template <typename>
struct always_false : std::false_type
{
};

template <typename AttrT>
struct attr_traits;

template <typename T, typename HandleT>
struct attr_traits<std::shared_ptr<rxmesh::Attribute<T, HandleT>>>
{
    using value_type  = T;
    using handle_type = HandleT;
};

template <typename T>
DType dtype_for()
{
    if constexpr (std::is_same_v<T, float>) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DType::Int32;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DType::Int8;
    } else {
        static_assert(always_false<T>::value, "Unsupported PyRXMesh dtype");
    }
}

template <typename HandleT>
ElementKind element_kind_for()
{
    if constexpr (std::is_same_v<HandleT, rxmesh::VertexHandle>) {
        return ElementKind::Vertex;
    } else if constexpr (std::is_same_v<HandleT, rxmesh::EdgeHandle>) {
        return ElementKind::Edge;
    } else if constexpr (std::is_same_v<HandleT, rxmesh::FaceHandle>) {
        return ElementKind::Face;
    } else {
        static_assert(always_false<HandleT>::value,
                      "Unsupported PyRXMesh element kind");
    }
}

DType parse_dtype(const std::string& dtype)
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

const char* dtype_name(DType dtype)
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

const char* element_kind_name(ElementKind element_kind)
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

rxmesh::locationT parse_location(int location)
{
    return static_cast<rxmesh::locationT>(location);
}

rxmesh::layoutT parse_layout(int layout)
{
    return static_cast<rxmesh::layoutT>(layout);
}

void ensure_polyscope_available()
{
#if !USE_POLYSCOPE
    throw std::runtime_error(
        "RXMesh was built with RX_USE_POLYSCOPE=OFF; visualization is not available.");
#endif
}

void show_polyscope()
{
    ensure_polyscope_available();
#if USE_POLYSCOPE
    polyscope::show();
#endif
}

py::module_ add_constants_module(py::module_& parent,
                                 const char*  name,
                                 const char*  doc)
{
    return parent.def_submodule(name, doc);
}

void destroy_mesh_capsule(PyObject* capsule)
{
    auto* data = static_cast<pyrxmesh::MeshCapsule*>(
        PyCapsule_GetPointer(capsule, pyrxmesh::mesh_capsule_name));
    if (data) {
        delete data;
    } else {
        PyErr_Clear();
    }
}

void destroy_attribute_capsule(PyObject* capsule)
{
    auto* data = static_cast<pyrxmesh::AttributeCapsule*>(
        PyCapsule_GetPointer(capsule, pyrxmesh::attribute_capsule_name));
    if (data) {
        delete data;
    } else {
        PyErr_Clear();
    }
}

template <typename T, typename HandleT>
uint32_t element_count(const rxmesh::RXMeshStatic& mesh)
{
    if constexpr (std::is_same_v<HandleT, rxmesh::VertexHandle>) {
        return mesh.get_num_vertices();
    } else if constexpr (std::is_same_v<HandleT, rxmesh::EdgeHandle>) {
        return mesh.get_num_edges();
    } else if constexpr (std::is_same_v<HandleT, rxmesh::FaceHandle>) {
        return mesh.get_num_faces();
    } else {
        static_assert(always_false<HandleT>::value,
                      "Unsupported PyRXMesh element kind");
    }
}

template <typename T, typename HandleT>
void ensure_host_readable(rxmesh::Attribute<T, HandleT>& attr)
{
    if (attr.is_host_allocated()) {
        return;
    }
    if (!attr.is_device_allocated()) {
        throw std::runtime_error(
            "Attribute is not allocated on HOST or DEVICE, so it cannot be copied to Python.");
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
            "Attribute is not allocated on HOST or DEVICE, so it cannot be populated from Python.");
    }
    attr.move(rxmesh::DEVICE, rxmesh::HOST);
}

template <typename T, typename HandleT>
py::array_t<T> attr_to_numpy(rxmesh::Attribute<T, HandleT>& attr,
                             const rxmesh::RXMeshStatic&    mesh,
                             rxmesh::locationT              source)
{
    if (source == rxmesh::DEVICE) {
        if (!attr.is_device_allocated()) {
            throw std::runtime_error(
                "Attribute is not allocated on DEVICE, so it cannot be copied from DEVICE.");
        }
        attr.move(rxmesh::DEVICE, rxmesh::HOST);
    } else {
        ensure_host_readable(attr);
    }

    const uint32_t rows = element_count<T, HandleT>(mesh);
    const uint32_t cols = attr.get_num_attributes();
    py::array_t<T> out({static_cast<py::ssize_t>(rows),
                        static_cast<py::ssize_t>(cols)});
    auto           view = out.template mutable_unchecked<2>();

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            view(i, j) = attr(i, j);
        }
    }
    return out;
}

template <typename T, typename HandleT>
void attr_from_numpy(rxmesh::Attribute<T, HandleT>& attr,
                     const rxmesh::RXMeshStatic&    mesh,
                     py::array                      values,
                     rxmesh::locationT              target)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> typed(values);
    const py::buffer_info info = typed.request();

    const uint32_t rows = element_count<T, HandleT>(mesh);
    const uint32_t cols = attr.get_num_attributes();

    if (info.ndim != 1 && info.ndim != 2) {
        throw std::invalid_argument(
            "Attribute.from_numpy() expects a 1D or 2D array.");
    }
    if (static_cast<uint32_t>(info.shape[0]) != rows) {
        throw std::invalid_argument(
            "Attribute.from_numpy() row count does not match the mesh element count.");
    }
    if (info.ndim == 1 && cols != 1) {
        throw std::invalid_argument(
            "Attribute.from_numpy() received a 1D array for a multi-column attribute.");
    }
    if (info.ndim == 2 && static_cast<uint32_t>(info.shape[1]) != cols) {
        throw std::invalid_argument(
            "Attribute.from_numpy() column count does not match the attribute dimension.");
    }

    ensure_host_writable(attr);

    if (info.ndim == 1) {
        auto view = typed.template unchecked<1>();
        for (uint32_t i = 0; i < rows; ++i) {
            attr(i, 0) = view(i);
        }
    } else {
        auto view = typed.template unchecked<2>();
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                attr(i, j) = view(i, j);
            }
        }
    }

    if ((target & rxmesh::DEVICE) == rxmesh::DEVICE) {
        attr.move(rxmesh::HOST, rxmesh::DEVICE);
    }
}

struct PyAttribute
{
    PyAttribute(std::shared_ptr<rxmesh::RXMeshStatic> owner,
                AttributeVariant                      attr)
        : mesh_owner(std::move(owner)), attribute(std::move(attr))
    {
    }

    std::shared_ptr<rxmesh::RXMeshStatic> mesh_owner;
    AttributeVariant                      attribute;

    const char* name() const
    {
        return std::visit(
            [](const auto& attr) -> const char* { return attr->get_name(); },
            attribute);
    }

    DType dtype() const
    {
        return std::visit(
            [](const auto& attr) -> DType {
                using AttrT = std::decay_t<decltype(attr)>;
                return dtype_for<typename attr_traits<AttrT>::value_type>();
            },
            attribute);
    }

    ElementKind element_kind() const
    {
        return std::visit(
            [](const auto& attr) -> ElementKind {
                using AttrT = std::decay_t<decltype(attr)>;
                return element_kind_for<typename attr_traits<AttrT>::handle_type>();
            },
            attribute);
    }

    uint32_t dim() const
    {
        return std::visit(
            [](const auto& attr) -> uint32_t {
                return attr->get_num_attributes();
            },
            attribute);
    }

    int allocated() const
    {
        return std::visit(
            [](const auto& attr) -> int {
                return static_cast<int>(attr->get_allocated());
            },
            attribute);
    }

    int layout() const
    {
        return std::visit(
            [](const auto& attr) -> int {
                return static_cast<int>(attr->get_layout());
            },
            attribute);
    }

    void reset(py::object value, int location)
    {
        const rxmesh::locationT loc = parse_location(location);
        std::visit(
            [&](const auto& attr) {
                using AttrT = std::decay_t<decltype(attr)>;
                using T     = typename attr_traits<AttrT>::value_type;
                attr->reset(value.cast<T>(), loc);
            },
            attribute);
    }

    void move(int source, int target)
    {
        std::visit(
            [&](const auto& attr) {
                attr->move(parse_location(source), parse_location(target));
            },
            attribute);
    }

    py::array to_numpy(int source)
    {
        const rxmesh::locationT src = parse_location(source);
        return std::visit(
            [&](const auto& attr) -> py::array {
                using AttrT   = std::decay_t<decltype(attr)>;
                using T       = typename attr_traits<AttrT>::value_type;
                using HandleT = typename attr_traits<AttrT>::handle_type;
                return attr_to_numpy<T, HandleT>(*attr, *mesh_owner, src);
            },
            attribute);
    }

    void from_numpy(py::array values, int target)
    {
        const rxmesh::locationT dst = parse_location(target);
        std::visit(
            [&](const auto& attr) {
                using AttrT   = std::decay_t<decltype(attr)>;
                using T       = typename attr_traits<AttrT>::value_type;
                using HandleT = typename attr_traits<AttrT>::handle_type;
                attr_from_numpy<T, HandleT>(*attr, *mesh_owner, values, dst);
            },
            attribute);
    }

    py::capsule capsule()
    {
        auto* capsule_data = new pyrxmesh::AttributeCapsule();
        capsule_data->abi_version    = plugin_abi_version;
        capsule_data->build_config   = build_config_tag;
        capsule_data->element_kind = static_cast<pyrxmesh::ElementKind>(
            static_cast<uint32_t>(element_kind()));
        capsule_data->dtype =
            static_cast<pyrxmesh::DType>(static_cast<uint32_t>(dtype()));
        capsule_data->num_attributes = dim();
        capsule_data->attribute      = std::visit(
            [](const auto& attr) -> rxmesh::AttributeBase* { return attr.get(); },
            attribute);

        return py::capsule(capsule_data,
                           pyrxmesh::attribute_capsule_name,
                           destroy_attribute_capsule);
    }
};

template <rxmesh::Op op>
struct DummyQueryLambda
{
    __device__ void operator()(typename rxmesh::InputHandle<op>::type,
                               typename rxmesh::IteratorType<op>::type) const
    {
    }
};

template <uint32_t blockThreads>
const void* dummy_query_kernel(rxmesh::Op op)
{
    switch (op) {
        case rxmesh::Op::VV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VV,
                                             DummyQueryLambda<rxmesh::Op::VV>>);
        case rxmesh::Op::VE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VE,
                                             DummyQueryLambda<rxmesh::Op::VE>>);
        case rxmesh::Op::VF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::VF,
                                             DummyQueryLambda<rxmesh::Op::VF>>);
        case rxmesh::Op::FV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FV,
                                             DummyQueryLambda<rxmesh::Op::FV>>);
        case rxmesh::Op::FE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FE,
                                             DummyQueryLambda<rxmesh::Op::FE>>);
        case rxmesh::Op::FF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::FF,
                                             DummyQueryLambda<rxmesh::Op::FF>>);
        case rxmesh::Op::EV:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EV,
                                             DummyQueryLambda<rxmesh::Op::EV>>);
        case rxmesh::Op::EE:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EE,
                                             DummyQueryLambda<rxmesh::Op::EE>>);
        case rxmesh::Op::EF:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EF,
                                             DummyQueryLambda<rxmesh::Op::EF>>);
        case rxmesh::Op::EVDiamond:
            return reinterpret_cast<const void*>(
                rxmesh::detail::query_kernel<blockThreads,
                                             rxmesh::Op::EVDiamond,
                                             DummyQueryLambda<rxmesh::Op::EVDiamond>>);
        default:
            throw std::invalid_argument(
                "Unsupported PyRXMesh query op for plugin launch.");
    }
}

template <typename HandleT, typename T>
std::shared_ptr<PyAttribute> make_attribute(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
    const std::string&                           name,
    uint32_t                                     dim,
    rxmesh::locationT                            location,
    rxmesh::layoutT                              layout)
{
    return std::make_shared<PyAttribute>(
        mesh, mesh->template add_attribute<T, HandleT>(name, dim, location, layout));
}

template <typename HandleT>
std::shared_ptr<PyAttribute> add_typed_attribute(
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
            return make_attribute<HandleT, float>(
                mesh, name, dim, loc, mem_layout);
        case DType::Float64:
            return make_attribute<HandleT, double>(
                mesh, name, dim, loc, mem_layout);
        case DType::Int32:
            return make_attribute<HandleT, int32_t>(
                mesh, name, dim, loc, mem_layout);
        case DType::Int8:
            return make_attribute<HandleT, int8_t>(
                mesh, name, dim, loc, mem_layout);
        default:
            throw std::invalid_argument("Unsupported RXMesh attribute dtype.");
    }
}

std::shared_ptr<PyAttribute> input_vertex_coordinates(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh)
{
    return std::make_shared<PyAttribute>(mesh, mesh->get_input_vertex_coordinates());
}

template <uint32_t blockThreads>
void fill_plugin_launch_box(rxmesh::RXMeshStatic*      mesh,
                            rxmesh::Op                 op,
                            const void*,
                            bool                       oriented,
                            pyrxmesh::PluginLaunchBox* out)
{
    rxmesh::LaunchBox<blockThreads> launch_box;
    mesh->prepare_launch_box(
        {op}, launch_box, dummy_query_kernel<blockThreads>(op), oriented);

    out->blocks                   = launch_box.blocks;
    out->num_threads              = launch_box.num_threads;
    out->num_registers_per_thread = launch_box.num_registers_per_thread;
    out->smem_bytes_dyn           = launch_box.smem_bytes_dyn;
    out->smem_bytes_static        = launch_box.smem_bytes_static;
    out->local_mem_per_thread     = launch_box.local_mem_per_thread;
}

void prepare_plugin_launch_box(rxmesh::RXMeshStatic* mesh,
                               rxmesh::Op            op,
                               uint32_t              block_threads,
                               const void*           kernel,
                               bool                  oriented,
                               void*                 launch_box)
{
    auto* out = static_cast<pyrxmesh::PluginLaunchBox*>(launch_box);
    switch (block_threads) {
        case 128:
            fill_plugin_launch_box<128>(mesh, op, kernel, oriented, out);
            return;
        case 256:
            fill_plugin_launch_box<256>(mesh, op, kernel, oriented, out);
            return;
        case 384:
            fill_plugin_launch_box<384>(mesh, op, kernel, oriented, out);
            return;
        case 512:
            fill_plugin_launch_box<512>(mesh, op, kernel, oriented, out);
            return;
        case 768:
            fill_plugin_launch_box<768>(mesh, op, kernel, oriented, out);
            return;
        case 1024:
            fill_plugin_launch_box<1024>(mesh, op, kernel, oriented, out);
            return;
        default:
            throw std::invalid_argument(
                "Unsupported PyRXMesh query block size. Use one of 128, 256, "
                "384, 512, 768, or 1024.");
    }
}

py::capsule mesh_capsule(const std::shared_ptr<rxmesh::RXMeshStatic>& mesh)
{
    auto* capsule_data             = new pyrxmesh::MeshCapsule();
    capsule_data->abi_version      = plugin_abi_version;
    capsule_data->build_config     = build_config_tag;
    capsule_data->mesh             = mesh.get();
    capsule_data->context          = &mesh->get_context();
    capsule_data->prepare_launch_box = &prepare_plugin_launch_box;

    return py::capsule(
        capsule_data, pyrxmesh::mesh_capsule_name, destroy_mesh_capsule);
}

}  // namespace

PYBIND11_MODULE(_rxmesh, m)
{
    m.doc() = "Python bindings for RXMesh";

    m.def("abi_version",
          []() { return plugin_abi_version; },
          "Return the PyRXMesh plugin ABI version.");
    m.def("build_config_tag",
          []() { return std::string(build_config_tag); },
          "Return the RXMesh/PyRXMesh build configuration tag.");

    py::module_ location = add_constants_module(
        m, "Location", "RXMesh memory location bitmask constants.");
    location.attr("NONE")   = py::int_(static_cast<int>(rxmesh::LOCATION_NONE));
    location.attr("HOST")   = py::int_(static_cast<int>(rxmesh::HOST));
    location.attr("DEVICE") = py::int_(static_cast<int>(rxmesh::DEVICE));
    location.attr("ALL")    = py::int_(static_cast<int>(rxmesh::LOCATION_ALL));

    py::module_ layout = add_constants_module(
        m, "Layout", "RXMesh attribute memory layout constants.");
    layout.attr("AoS") = py::int_(static_cast<int>(rxmesh::AoS));
    layout.attr("SoA") = py::int_(static_cast<int>(rxmesh::SoA));

    py::enum_<ElementKind>(m, "ElementKind")
        .value("Vertex", ElementKind::Vertex)
        .value("Edge", ElementKind::Edge)
        .value("Face", ElementKind::Face);

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int8", DType::Int8);

    py::enum_<rxmesh::Op>(m, "Op")
        .value("INVALID", rxmesh::Op::INVALID)
        .value("V", rxmesh::Op::V)
        .value("E", rxmesh::Op::E)
        .value("F", rxmesh::Op::F)
        .value("VV", rxmesh::Op::VV)
        .value("VE", rxmesh::Op::VE)
        .value("VF", rxmesh::Op::VF)
        .value("FV", rxmesh::Op::FV)
        .value("FE", rxmesh::Op::FE)
        .value("FF", rxmesh::Op::FF)
        .value("EV", rxmesh::Op::EV)
        .value("EE", rxmesh::Op::EE)
        .value("EF", rxmesh::Op::EF)
        .value("EVDiamond", rxmesh::Op::EVDiamond);

    m.def("init",
          [](int device_id) { rxmesh::rx_init(device_id); },
          py::arg("device_id") = 0,
          "Initialize RXMesh logging and select a CUDA device.");

    m.def("show", &show_polyscope, "Open the Polyscope viewer.");

    py::class_<PyAttribute, std::shared_ptr<PyAttribute>>(m, "Attribute")
        .def_property_readonly("name", &PyAttribute::name)
        .def_property_readonly(
            "dtype",
            [](const PyAttribute& self) {
                return std::string(dtype_name(self.dtype()));
            })
        .def_property_readonly(
            "element_kind",
            [](const PyAttribute& self) {
                return std::string(element_kind_name(self.element_kind()));
            })
        .def_property_readonly("dim", &PyAttribute::dim)
        .def_property_readonly("allocated", &PyAttribute::allocated)
        .def_property_readonly("layout", &PyAttribute::layout)
        .def("reset",
             &PyAttribute::reset,
             py::arg("value"),
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL))
        .def("move",
             &PyAttribute::move,
             py::arg("source"),
             py::arg("target"))
        .def("to_numpy",
             &PyAttribute::to_numpy,
             py::arg("source") = static_cast<int>(rxmesh::DEVICE),
             "Copy this attribute to a NumPy array in input/global element order.")
        .def("from_numpy",
             &PyAttribute::from_numpy,
             py::arg("values"),
             py::arg("target") = static_cast<int>(rxmesh::LOCATION_ALL),
             "Copy a NumPy-compatible array into this attribute.")
        .def("__rxmesh_capsule__",
             &PyAttribute::capsule,
             "Return a low-level capsule for compiled PyRXMesh plugins.");

    py::class_<rxmesh::RXMeshStatic, std::shared_ptr<rxmesh::RXMeshStatic>>(
        m, "RXMeshStatic")
        .def(py::init<const std::string,
                      const std::string,
                      const uint32_t,
                      const float,
                      const float,
                      const float>(),
             py::arg("file_path"),
             py::arg("patcher_file")             = "",
             py::arg("patch_size")               = 512,
             py::arg("capacity_factor")          = 1.0f,
             py::arg("patch_alloc_factor")       = 1.0f,
             py::arg("lp_hashtable_load_factor") = 0.8f,
             "Load a static triangle mesh from an OBJ file.")
        .def_property_readonly(
            "num_vertices",
            [](const rxmesh::RXMeshStatic& self) { return self.get_num_vertices(); })
        .def_property_readonly(
            "num_edges",
            [](const rxmesh::RXMeshStatic& self) { return self.get_num_edges(); })
        .def_property_readonly(
            "num_faces",
            [](const rxmesh::RXMeshStatic& self) { return self.get_num_faces(); })
        .def_property_readonly(
            "num_patches",
            [](const rxmesh::RXMeshStatic& self) { return self.get_num_patches(); })
        .def_property_readonly(
            "num_components",
            [](const rxmesh::RXMeshStatic& self) { return self.get_num_components(); })
        .def("is_closed",
             [](const rxmesh::RXMeshStatic& self) { return self.is_closed(); })
        .def("is_edge_manifold",
             [](const rxmesh::RXMeshStatic& self) { return self.is_edge_manifold(); })
        .def("show", &show_polyscope, "Open the Polyscope viewer for this mesh.")
        .def("input_vertex_coordinates",
             &input_vertex_coordinates,
             "Return the input vertex coordinate attribute.")
        .def("add_vertex_attribute",
             &add_typed_attribute<rxmesh::VertexHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(rxmesh::SoA),
             "Add a typed vertex attribute.")
        .def("add_edge_attribute",
             &add_typed_attribute<rxmesh::EdgeHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(rxmesh::SoA),
             "Add a typed edge attribute.")
        .def("add_face_attribute",
             &add_typed_attribute<rxmesh::FaceHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(rxmesh::LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(rxmesh::SoA),
             "Add a typed face attribute.")
        .def("__rxmesh_capsule__",
             &mesh_capsule,
             "Return a low-level capsule for compiled PyRXMesh plugins.");
}
