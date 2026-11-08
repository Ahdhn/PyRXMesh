#include "bindings/plugin_launch.h"
#include "bindings/dlpack_utils.h"
#include "bindings/py_attribute.h"

namespace pyrxmesh_py {

namespace {

// Native attributes outlive their Python wrappers while registered on the mesh.
// Keep borrowed buffers with the native mesh, not just with a Python wrapper.
struct MeshDeleter
{
    std::unordered_map<std::string, std::shared_ptr<void>> attribute_owners;

    void operator()(rxmesh::RXMeshStatic* mesh) const { delete mesh; }
};

template <typename... Args>
std::shared_ptr<rxmesh::RXMeshStatic> make_mesh(Args&&... args)
{
    return {new rxmesh::RXMeshStatic(std::forward<Args>(args)...), MeshDeleter{}};
}

std::shared_ptr<rxmesh::RXMeshStatic> rxmesh_static_from_file(
    const std::string& file_path,
    const std::string& patcher_file,
    uint32_t           patch_size,
    float              capacity_factor,
    float              patch_alloc_factor,
    float              lp_hashtable_load_factor)
{
    return make_mesh(file_path,
                                                  patcher_file,
                                                  patch_size,
                                                  capacity_factor,
                                                  patch_alloc_factor,
                                                  lp_hashtable_load_factor,
                                                  rxmesh::SoA);
}

std::shared_ptr<rxmesh::RXMeshStatic> rxmesh_static_from_files(
    const std::vector<std::string>& file_paths,
    uint32_t                        patch_size)
{
    return make_mesh(
        file_paths, patch_size, rxmesh::SoA);
}

std::shared_ptr<rxmesh::RXMeshStatic> rxmesh_static_from_arrays(
    py::array_t<rx_coord_t, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast>   faces,
    const std::string& patcher_file,
    uint32_t           patch_size,
    float              capacity_factor,
    float              patch_alloc_factor,
    float              lp_hashtable_load_factor)
{
    if (vertices.ndim() != 2 || vertices.shape(1) != 3) {
        throw py::value_error("vertices must have shape (n, 3)");
    }
    if (faces.ndim() != 2 || faces.shape(1) != 3) {
        throw py::value_error("faces must have shape (m, 3)");
    }

    const auto vertices_view = vertices.unchecked<2>();
    const auto faces_view    = faces.unchecked<2>();
    std::vector<std::vector<rx_coord_t>> vertex_list(
        static_cast<size_t>(vertices.shape(0)), std::vector<rx_coord_t>(3));
    std::vector<std::vector<uint32_t>> face_list(
        static_cast<size_t>(faces.shape(0)), std::vector<uint32_t>(3));

    for (py::ssize_t i = 0; i < vertices.shape(0); ++i) {
        for (py::ssize_t j = 0; j < 3; ++j) {
            vertex_list[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                vertices_view(i, j);
        }
    }
    for (py::ssize_t i = 0; i < faces.shape(0); ++i) {
        for (py::ssize_t j = 0; j < 3; ++j) {
            face_list[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                faces_view(i, j);
        }
    }

    auto mesh =
        make_mesh(face_list,
                                               patcher_file,
                                               patch_size,
                                               capacity_factor,
                                               patch_alloc_factor,
                                               lp_hashtable_load_factor);
    mesh->add_vertex_coordinates(vertex_list, "", rxmesh::SoA);
    return mesh;
}

template <typename HandleT>
py::array_t<uint64_t> handles(rxmesh::RXMeshStatic& mesh)
{
    py::array_t<uint64_t> out(
        static_cast<py::ssize_t>(mesh.get_num_elements<HandleT>()));

    auto view = out.mutable_unchecked<1>();

    mesh.for_each<HandleT>(rxmesh::HOST, [&](const HandleT h) {
        view(mesh.linear_id(h)) = h.unique_id();
    });
    return out;
}

template <typename HandleT>
void for_each_handle(rxmesh::RXMeshStatic& mesh, py::function callback)
{
    mesh.for_each<HandleT>(
        rxmesh::HOST, [&](const HandleT h) { callback(h); }, nullptr, false);
}

template <typename HandleT>
auto global_id_of(const rxmesh::RXMeshStatic& mesh, HandleT h)
{
    return mesh.map_to_global(h);
}

template <typename HandleT>
auto linear_id_of(rxmesh::RXMeshStatic& mesh, HandleT h)
{
    return mesh.linear_id(h);
}

template <bool LinearToGlobal, typename HandleT>
py::array_t<uint32_t> element_order_map_typed(rxmesh::RXMeshStatic& mesh)
{
    py::array_t<uint32_t> out(
        static_cast<py::ssize_t>(mesh.get_num_elements<HandleT>()));
    uint32_t* values = out.mutable_data();
    mesh.for_each<HandleT>(
        rxmesh::HOST,
        [&](const HandleT h) {
            const uint32_t linear = mesh.linear_id(h);
            const uint32_t global = mesh.map_to_global(h);
            values[LinearToGlobal ? linear : global] =
                LinearToGlobal ? global : linear;
        },
        nullptr,
        false);
    return out;
}

template <bool LinearToGlobal>
py::array_t<uint32_t> element_order_map(rxmesh::RXMeshStatic& mesh,
                                        ElementKind           kind)
{
    switch (kind) {
        case ElementKind::Vertex:
            return element_order_map_typed<LinearToGlobal,
                                           rxmesh::VertexHandle>(mesh);
        case ElementKind::Edge:
            return element_order_map_typed<LinearToGlobal, rxmesh::EdgeHandle>(
                mesh);
        case ElementKind::Face:
            return element_order_map_typed<LinearToGlobal, rxmesh::FaceHandle>(
                mesh);
        default:
            throw std::invalid_argument(
                "Element order map requires ElementKind.Vertex, Edge, or "
                "Face.");
    }
}

bool use_global_order(const std::string& order)
{
    if (order == "linear") {
        return false;
    }
    if (order == "global") {
        return true;
    }
    throw std::invalid_argument("order must be 'linear' or 'global'");
}

py::array_t<rx_coord_t> vertices(const rxmesh::RXMeshStatic& mesh,
                                 const std::string&          order)
{
    const bool global_order = use_global_order(order);

    py::array_t<rx_coord_t> out(
        {static_cast<py::ssize_t>(mesh.get_num_vertices()),
         static_cast<py::ssize_t>(3)});
    mesh.get_input_vertex_coordinates(out.mutable_data(), global_order);
    return out;
}

py::array_t<uint32_t> faces(const rxmesh::RXMeshStatic& mesh,
                            const std::string&          order)
{
    const bool global_order = use_global_order(order);

    py::array_t<uint32_t> out({static_cast<py::ssize_t>(mesh.get_num_faces()),
                               static_cast<py::ssize_t>(3)});
    mesh.create_face_list(out.mutable_data(), global_order);
    return out;
}

py::array_t<uint32_t> edges(const rxmesh::RXMeshStatic& mesh,
                            const std::string&          order)
{
    const bool global_order = use_global_order(order);

    py::array_t<uint32_t> out({static_cast<py::ssize_t>(mesh.get_num_edges()),
                               static_cast<py::ssize_t>(2)});
    mesh.create_edge_list(out.mutable_data(), global_order);
    return out;
}

py::array polyscope_edge_permutation(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh)
{
    const auto& permutation = mesh->get_edge_permutation();
    py::array   out =
        py::array_t<uint32_t>({static_cast<py::ssize_t>(permutation.size())},
                              {static_cast<py::ssize_t>(sizeof(uint32_t))},
                              permutation.data(),
                              py::cast(mesh));
    out.attr("setflags")(py::arg("write") = false);
    return out;
}

py::tuple patch_size_stats(rxmesh::RXMeshStatic& mesh)
{
    uint32_t min_p = 0;
    uint32_t max_p = 0;
    uint32_t avg_p = 0;
    mesh.get_max_min_avg_patch_size(min_p, max_p, avg_p);
    return py::make_tuple(min_p, max_p, avg_p);
}

py::tuple bounding_box(rxmesh::RXMeshStatic& mesh)
{
    glm::vec3 lower;
    glm::vec3 upper;
    mesh.bounding_box(lower, upper);
    return py::make_tuple(vec3_to_numpy(lower), vec3_to_numpy(upper));
}

void scale(rxmesh::RXMeshStatic& mesh,
           const py::sequence&   lower,
           const py::sequence&   upper)
{
    mesh.scale(sequence_to_fvec3(lower, "lower"),
               sequence_to_fvec3(upper, "upper"));
}

bool has_attribute(rxmesh::RXMeshStatic& mesh, const std::string& name)
{
    return mesh.does_attribute_exist(name);
}

void remove_attribute(const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
                      const std::string& name)
{
    auto* storage = std::get_deleter<MeshDeleter>(mesh);
    if (storage && storage->attribute_owners.count(name)) {
        // An external buffer must not be returned to its allocator while a
        // previously launched kernel can still use it.
        using namespace rxmesh;
        CUDA_ERROR(cudaDeviceSynchronize());
        mesh->remove_attribute(name);
        storage->attribute_owners.erase(name);
    } else {
        mesh->remove_attribute(name);
    }
}

template <typename HandleT>
py::object add_attribute_from_dlpack_view(
    const std::shared_ptr<rxmesh::RXMeshStatic>& mesh,
    const std::string& name, py::object source)
{
    auto* storage = std::get_deleter<MeshDeleter>(mesh);
    if (!storage) {
        throw std::invalid_argument("Tensor views require a mesh created by PyRXMesh.");
    }
    if (mesh->does_attribute_exist(name)) {
        throw std::invalid_argument("An attribute with this name already exists.");
    }
    auto capsule = dlpack_util::acquire_capsule(
        std::move(source), "add_attribute_from_dlpack_view()");
    auto* managed = dlpack_util::extract_managed(capsule);
    if (!managed) {
        throw std::invalid_argument("Expected an unconsumed DLPack tensor.");
    }
    // Transfer capsule ownership to RAII immediately, including error paths.
    if (PyCapsule_SetName(capsule.ptr(), "used_dltensor") != 0) {
        throw py::error_already_set();
    }
    auto owner = std::shared_ptr<void>(managed, [](void* pointer) {
        auto* tensor = static_cast<dlpack::DLManagedTensor*>(pointer);
        if (tensor->deleter) {
            tensor->deleter(tensor);
        }
    });
    const auto& tensor = managed->dl_tensor;
    int device = 0;
    using namespace rxmesh;
    CUDA_ERROR(cudaGetDevice(&device));
    if (tensor.device.device_type != dlpack::kDLCUDA ||
        tensor.device.device_id != device) {
        throw std::invalid_argument("Expected a CUDA tensor on the current RXMesh device.");
    }
    if (tensor.ndim != 2 ||
        tensor.shape[0] != mesh->get_num_elements<HandleT>() ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0 ||
        tensor.shape[1] > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Expected shape (mesh element count, channels).");
    }
    const int64_t rows = tensor.shape[0], cols = tensor.shape[1];
    const int64_t stride0 = tensor.strides ? tensor.strides[0] : cols;
    const int64_t stride1 = tensor.strides ? tensor.strides[1] : 1;
    // Singleton dimensions do not constrain the corresponding stride. In
    // particular, ordinary contiguous (N, 1) tensors already have SoA storage.
    if ((rows > 1 && stride0 != 1) || (cols > 1 && stride1 != rows)) {
        throw std::invalid_argument("Zero-copy attribute views require SoA strides (1, N); no copy is performed.");
    }
    return dispatch_dtype_str(dlpack_util::dtype_to_name(tensor.dtype), [&](auto tag) {
        using T = typename decltype(tag)::type;
        // Metadata only: no temporary attribute data allocation is needed.
        auto attr = mesh->template add_attribute<T, HandleT>(
            name, static_cast<uint32_t>(cols), LOCATION_NONE, SoA);
        auto* pointer = reinterpret_cast<T*>(
            static_cast<char*>(tensor.data) + tensor.byte_offset);
        if (!attr->attach_device_buffer(pointer, size_t(rows) * size_t(cols) * sizeof(T))) {
            mesh->remove_attribute(name);
            throw std::invalid_argument("Unable to attach the tensor's CUDA buffer.");
        }
        auto out = std::make_shared<PyAttribute<T, HandleT>>(mesh, attr);
        out->external_owner = owner;
        storage->attribute_owners.emplace(name, std::move(owner));
        return py::cast(out);
    });
}

void export_obj(rxmesh::RXMeshStatic& mesh,
                const std::string&    filename,
                PyAttributeBase&      coords)
{
    if (coords.mesh_owner.get() != &mesh) {
        throw std::invalid_argument(
            "RXMeshStatic.export_obj() requires a coordinate attribute from "
            "the same mesh.");
    }
    if (coords.element_kind() != ElementKind::Vertex) {
        throw std::invalid_argument(
            "RXMeshStatic.export_obj() coords must be a vertex attribute.");
    }
    if (coords.dim() < 3) {
        throw std::invalid_argument(
            "RXMeshStatic.export_obj() coords must have at least 3 columns.");
    }
    if (coords.dtype() != DType::Float32 && coords.dtype() != DType::Float64) {
        throw std::invalid_argument(
            "RXMeshStatic.export_obj() coords must have dtype float32 or "
            "float64.");
    }

    coords.export_obj(mesh, filename);
}

}  // namespace

void register_mesh(py::module_& m)
{
    using namespace rxmesh;

    py::class_<RXMeshStatic, std::shared_ptr<RXMeshStatic>>(
        m, "RXMeshStatic", py::dynamic_attr())
        .def(py::init(&rxmesh_static_from_file),
             py::arg("file_path"),
             py::arg("patcher_file")             = "",
             py::arg("patch_size")               = 512,
             py::arg("capacity_factor")          = 1.0f,
             py::arg("patch_alloc_factor")       = 1.0f,
             py::arg("lp_hashtable_load_factor") = 0.8f,
             "Load a static triangle mesh from an OBJ file.")
        .def(py::init(&rxmesh_static_from_arrays),
             py::arg("vertices"),
             py::arg("faces"),
             py::arg("patcher_file")             = "",
             py::arg("patch_size")               = 512,
             py::arg("capacity_factor")          = 1.0f,
             py::arg("patch_alloc_factor")       = 1.0f,
             py::arg("lp_hashtable_load_factor") = 0.8f,
             "Create a static triangle mesh from (n, 3) vertices and (m, 3) "
             "face arrays.")
        .def_static("from_files",
                    &rxmesh_static_from_files,
                    py::arg("file_paths"),
                    py::arg("patch_size") = 512,
                    "Load multiple OBJ files into one RXMeshStatic.")
        .def_property_readonly(
            "num_vertices",
            [](const RXMeshStatic& self) { return self.get_num_vertices(); })
        .def_property_readonly(
            "num_edges",
            [](const RXMeshStatic& self) { return self.get_num_edges(); })
        .def_property_readonly(
            "num_faces",
            [](const RXMeshStatic& self) { return self.get_num_faces(); })
        .def_property_readonly(
            "num_patches",
            [](const RXMeshStatic& self) { return self.get_num_patches(); })
        .def_property_readonly(
            "max_num_patches",
            [](const RXMeshStatic& self) { return self.get_max_num_patches(); })
        .def_property_readonly(
            "num_components",
            [](const RXMeshStatic& self) { return self.get_num_components(); })
        .def_property_readonly(
            "num_colors",
            [](const RXMeshStatic& self) { return self.get_num_colors(); })
        .def_property_readonly(
            "patch_size",
            [](const RXMeshStatic& self) { return self.get_patch_size(); })
        .def_property_readonly("input_max_valence",
                               [](const RXMeshStatic& self) {
                                   return self.get_input_max_valence();
                               })
        .def_property_readonly(
            "input_max_edge_incident_faces",
            [](const RXMeshStatic& self) {
                return self.get_input_max_edge_incident_faces();
            })
        .def_property_readonly(
            "input_max_face_adjacent_faces",
            [](const RXMeshStatic& self) {
                return self.get_input_max_face_adjacent_faces();
            })
        .def_property_readonly(
            "patching_time",
            [](const RXMeshStatic& self) { return self.get_patching_time(); })
        .def("is_closed", &RXMeshStatic::is_closed)
        .def("is_edge_manifold", &RXMeshStatic::is_edge_manifold)
        .def("patch_size_stats",
             &patch_size_stats,
             "Return (min_patch_size, max_patch_size, avg_patch_size).")
        .def(
            "input_vertex_coordinates",
            [](std::shared_ptr<RXMeshStatic> mesh) {
                auto attr = mesh->get_input_vertex_coordinates();
                return py::cast(std::make_shared<
                                PyAttribute<rx_coord_t, rxmesh::VertexHandle>>(
                    mesh, std::move(attr)));
            },
            "Return the input vertex coordinate attribute.")
        .def("vertices",
             &vertices,
             py::arg("order") = "linear",
             "Return input coordinates in RXMesh linear or global row "
             "order.")
        .def("faces",
             &faces,
             py::arg("order") = "linear",
             "Return faces in RXMesh linear or global order. The selected "
             "order applies to both face rows and vertex IDs.")
        .def("edges",
             &edges,
             py::arg("order") = "linear",
             "Return edges in RXMesh linear or global order. The selected "
             "order applies to both edge rows and vertex IDs.")
        .def("polyscope_edge_permutation",
             &polyscope_edge_permutation,
             "Return the read-only zero-copy permutation expected by "
             "Polyscope's set_edge_permutation(). Keep the mesh alive while "
             "the view is in use.")
        .def("bounding_box",
             &bounding_box,
             "Return (lower, upper) NumPy arrays for the mesh bounding box.")
        .def("scale",
             &scale,
             py::arg("lower"),
             py::arg("upper"),
             "Scale the mesh into the bounding box [lower, upper].")
        .def("save_patcher",
             &RXMeshStatic::save,
             py::arg("filename"),
             "Save RXMesh patching data to a file.")
        .def("export_obj",
             &export_obj,
             py::arg("filename"),
             py::arg("coords"),
             "Export the mesh to an OBJ file using a vertex coordinate "
             "attribute.")
        .def("vertex_handles", &handles<VertexHandle>)
        .def("edge_handles", &handles<EdgeHandle>)
        .def("face_handles", &handles<FaceHandle>)
        .def("for_each_vertex",
             &for_each_handle<VertexHandle>,
             py::arg("callback"))
        .def("for_each_edge", &for_each_handle<EdgeHandle>, py::arg("callback"))
        .def("for_each_face", &for_each_handle<FaceHandle>, py::arg("callback"))
        .def("global_id", &global_id_of<VertexHandle>, py::arg("handle"))
        .def("global_id", &global_id_of<EdgeHandle>, py::arg("handle"))
        .def("global_id", &global_id_of<FaceHandle>, py::arg("handle"))
        .def("linear_id", &linear_id_of<VertexHandle>, py::arg("handle"))
        .def("linear_id", &linear_id_of<EdgeHandle>, py::arg("handle"))
        .def("linear_id", &linear_id_of<FaceHandle>, py::arg("handle"))
        .def("linear_to_global",
             &element_order_map<true>,
             py::arg("element_kind"),
             "Return an owned uint32 map from RXMesh linear row IDs to "
             "map_to_global() IDs for the requested element kind.")
        .def("global_to_linear",
             &element_order_map<false>,
             py::arg("element_kind"),
             "Return an owned uint32 inverse map from map_to_global() IDs "
             "to RXMesh linear row IDs for the requested element kind.")
        .def("add_vertex_attribute",
             &add_typed_attribute<VertexHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = "all",
             py::arg("layout")   = "soa",
             "Add a typed vertex attribute.")
        .def("add_edge_attribute",
             &add_typed_attribute<EdgeHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = "all",
             py::arg("layout")   = "soa",
             "Add a typed edge attribute.")
        .def("add_face_attribute",
             &add_typed_attribute<FaceHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = "all",
             py::arg("layout")   = "soa",
             "Add a typed face attribute.")
        .def("add_vertex_attribute_from_dlpack_view",
             &add_attribute_from_dlpack_view<VertexHandle>,
             py::arg("name"), py::arg("source"),
             "Share a CUDA SoA tensor in RXMesh linear vertex order without copying.")
        .def("add_edge_attribute_from_dlpack_view",
             &add_attribute_from_dlpack_view<EdgeHandle>,
             py::arg("name"), py::arg("source"),
             "Share a CUDA SoA tensor in RXMesh linear edge order without copying.")
        .def("add_face_attribute_from_dlpack_view",
             &add_attribute_from_dlpack_view<FaceHandle>,
             py::arg("name"), py::arg("source"),
             "Share a CUDA SoA tensor in RXMesh linear face order without copying.")
        .def("add_attribute_like",
             &add_attribute_like,
             py::arg("name"),
             py::arg("other"),
             "Add a new attribute with the same element kind, dtype, "
             "dimension, allocation, and layout as another attribute.")
        .def(
            "attribute_names",
            [](RXMeshStatic& mesh) { return mesh.get_attribute_names(); },
            "Return names of attributes currently registered on this mesh.")
        .def("has_attribute",
             &has_attribute,
             py::arg("name"),
             "Return True when the mesh has an attribute with this name.")
        .def("remove_attribute",
             &remove_attribute,
             py::arg("name"),
             "Remove an attribute by name.")
        .def(
            "sparse_matrix",
            [](std::shared_ptr<RXMeshStatic> self,
               py::object                    op,
               std::string                   dtype) {
                return make_sparse_matrix_from_mesh(
                    std::move(self), parse_op(op), std::move(dtype));
            },
            py::arg("op")    = "vv",
            py::arg("dtype") = "float32",
            "Build an RXMesh-owned CSR sparse matrix for a mesh query op.")
        .def("__rxmesh_capsule__",
             &mesh_capsule,
             "Return a low-level capsule for compiled PyRXMesh plugins.");
}

}  // namespace pyrxmesh_py
