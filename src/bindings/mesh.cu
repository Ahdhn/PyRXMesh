#include "bindings/plugin_launch.h"
#include "bindings/py_attribute.h"

namespace pyrxmesh_py {

namespace {

std::shared_ptr<rxmesh::RXMeshStatic> rxmesh_static_from_file(
    const std::string& file_path,
    const std::string& patcher_file,
    uint32_t           patch_size,
    float              capacity_factor,
    float              patch_alloc_factor,
    float              lp_hashtable_load_factor)
{
    return std::make_shared<rxmesh::RXMeshStatic>(file_path,
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
    return std::make_shared<rxmesh::RXMeshStatic>(
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
        std::make_shared<rxmesh::RXMeshStatic>(face_list,
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

void remove_attribute(rxmesh::RXMeshStatic& mesh, const std::string& name)
{
    mesh.remove_attribute(name);
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
            "show", &show_polyscope, "Open the Polyscope viewer for this mesh.")
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
