#include "bindings/py_attribute.h"

namespace pyrxmesh_py {

namespace {

std::shared_ptr<rxmesh::RXMeshStatic> rxmesh_static_from_files(
    const std::vector<std::string>& file_paths,
    uint32_t                        patch_size)
{
    return std::make_shared<rxmesh::RXMeshStatic>(file_paths, patch_size);
}

py::array_t<uint64_t> vertex_handles(rxmesh::RXMeshStatic& mesh)
{

    py::array_t<uint64_t> out(
        static_cast<py::ssize_t>(mesh.get_num_vertices()));

    auto view = out.mutable_unchecked<1>();

    mesh.for_each_vertex(rxmesh::HOST, [&](const rxmesh::VertexHandle h) {
        view(mesh.linear_id(h)) = h.unique_id();
    });
    return out;
}

py::array_t<uint64_t> edge_handles(rxmesh::RXMeshStatic& mesh)
{
    py::array_t<uint64_t> out(static_cast<py::ssize_t>(mesh.get_num_edges()));

    auto view = out.mutable_unchecked<1>();

    mesh.for_each_edge(rxmesh::HOST, [&](const rxmesh::EdgeHandle h) {
        view(mesh.linear_id(h)) = h.unique_id();
    });
    return out;
}

py::array_t<uint64_t> face_handles(rxmesh::RXMeshStatic& mesh)
{
    py::array_t<uint64_t> out(static_cast<py::ssize_t>(mesh.get_num_faces()));

    auto view = out.mutable_unchecked<1>();

    mesh.for_each_face(rxmesh::HOST, [&](const rxmesh::FaceHandle h) {
        view(mesh.linear_id(h)) = h.unique_id();
    });
    return out;
}

py::array vertices(rxmesh::RXMeshStatic& mesh)
{
    auto attr = mesh.get_input_vertex_coordinates();
    ensure_host_readable(*attr);

    py::array_t<rx_coord_t> out(
        {static_cast<py::ssize_t>(attr->rows()), static_cast<py::ssize_t>(3)});
    auto view = out.mutable_unchecked<2>();

    for (uint32_t i = 0; i < attr->rows(); ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            view(i, j) = (*attr)(i, j);
        }
    }
    return out;
}

py::array_t<uint32_t> faces(const rxmesh::RXMeshStatic& mesh)
{
    std::vector<glm::uvec3> face_list;
    mesh.create_face_list(face_list);

    py::array_t<uint32_t> out({static_cast<py::ssize_t>(face_list.size()),
                               static_cast<py::ssize_t>(3)});
    auto                  view = out.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(face_list.size());
         ++i) {
        const auto& face = face_list[static_cast<size_t>(i)];
        view(i, 0)       = face[0];
        view(i, 1)       = face[1];
        view(i, 2)       = face[2];
    }
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

    py::class_<RXMeshStatic, std::shared_ptr<RXMeshStatic>>(m, "RXMeshStatic")
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
             "Return input vertex coordinates as a NumPy array copy.")
        .def("faces",
             &faces,
             "Return face vertex indices as a NumPy uint32 array.")
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
        .def("vertex_handles", &vertex_handles)
        .def("edge_handles", &edge_handles)
        .def("face_handles", &face_handles)
        .def(
            "for_each_vertex",
            [](RXMeshStatic& mesh, py::function callback) {
                mesh.for_each_vertex(
                    HOST,
                    [&](const VertexHandle h) { callback(h); },
                    nullptr,
                    false);
            },
            py::arg("callback"))
        .def(
            "for_each_edge",
            [](RXMeshStatic& mesh, py::function callback) {
                mesh.for_each_edge(
                    HOST,
                    [&](const EdgeHandle h) { callback(h); },
                    nullptr,
                    false);
            },
            py::arg("callback"))
        .def(
            "for_each_face",
            [](RXMeshStatic& mesh, py::function callback) {
                mesh.for_each_face(
                    HOST,
                    [&](const FaceHandle h) { callback(h); },
                    nullptr,
                    false);
            },
            py::arg("callback"))
        .def(
            "global_id",
            [](const RXMeshStatic& mesh, VertexHandle h) {
                return mesh.map_to_global(h);
            },
            py::arg("handle"))
        .def(
            "global_id",
            [](const RXMeshStatic& mesh, EdgeHandle h) {
                return mesh.map_to_global(h);
            },
            py::arg("handle"))
        .def(
            "global_id",
            [](const RXMeshStatic& mesh, FaceHandle h) {
                return mesh.map_to_global(h);
            },
            py::arg("handle"))
        .def(
            "linear_id",
            [](RXMeshStatic& mesh, const VertexHandle h) {
                return mesh.linear_id(h);
            },
            py::arg("handle"))
        .def(
            "linear_id",
            [](RXMeshStatic& mesh, const EdgeHandle h) {
                return mesh.linear_id(h);
            },
            py::arg("handle"))
        .def(
            "linear_id",
            [](RXMeshStatic& mesh, const FaceHandle h) {
                return mesh.linear_id(h);
            },
            py::arg("handle"))
        .def("add_vertex_attribute",
             &add_typed_attribute<VertexHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(SoA),
             "Add a typed vertex attribute.")
        .def("add_edge_attribute",
             &add_typed_attribute<EdgeHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(SoA),
             "Add a typed edge attribute.")
        .def("add_face_attribute",
             &add_typed_attribute<FaceHandle>,
             py::arg("name"),
             py::arg("dtype")    = "float32",
             py::arg("dim")      = 1,
             py::arg("location") = static_cast<int>(LOCATION_ALL),
             py::arg("layout")   = static_cast<int>(SoA),
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
            [](std::shared_ptr<RXMeshStatic> self, Op op, std::string dtype) {
                return make_sparse_matrix_from_mesh(
                    std::move(self), op, std::move(dtype));
            },
            py::arg("op")    = Op::VV,
            py::arg("dtype") = "float32",
            "Build an RXMesh-owned CSR sparse matrix for a mesh query op.")
        .def("__rxmesh_capsule__",
             &mesh_capsule,
             "Return a low-level capsule for compiled PyRXMesh plugins.");
}

}  // namespace pyrxmesh_py
