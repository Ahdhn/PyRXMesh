#include "bindings/common.h"

#include "rxmesh/geometry_factory.h"

namespace pyrxmesh_py {

namespace {

template <typename T>
py::tuple create_plane_typed(uint32_t     nx,
                             uint32_t     ny,
                             int          plane,
                             T            dx,
                             bool         with_cross_diagonal,
                             py::sequence low_corner)
{
    if (nx == 0 || ny == 0) {
        throw std::invalid_argument(
            "create_plane() nx and ny must be positive.");
    }
    if (plane < 0 || plane > 2) {
        throw std::invalid_argument("create_plane() plane must be 0, 1, or 2.");
    }
    if (py::len(low_corner) != 3) {
        throw std::invalid_argument(
            "create_plane() low_corner must contain exactly 3 values.");
    }

    rxmesh::vec3<T> corner(low_corner[0].cast<T>(),
                           low_corner[1].cast<T>(),
                           low_corner[2].cast<T>());

    std::vector<std::vector<T>>        verts;
    std::vector<std::vector<uint32_t>> tris;
    rxmesh::create_plane<T>(
        verts, tris, nx, ny, plane, dx, with_cross_diagonal, corner);

    py::array_t<T> vertices(
        {static_cast<py::ssize_t>(verts.size()), static_cast<py::ssize_t>(3)});
    py::array_t<uint32_t> faces(
        {static_cast<py::ssize_t>(tris.size()), static_cast<py::ssize_t>(3)});

    auto vertices_view = vertices.template mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(verts.size()); ++i) {
        for (py::ssize_t j = 0; j < 3; ++j) {
            vertices_view(i, j) =
                verts[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    auto faces_view = faces.template mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tris.size()); ++i) {
        for (py::ssize_t j = 0; j < 3; ++j) {
            faces_view(i, j) =
                tris[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return py::make_tuple(vertices, faces);
}

py::tuple create_plane(uint32_t     nx,
                       uint32_t     ny,
                       int          plane,
                       double       dx,
                       bool         with_cross_diagonal,
                       py::sequence low_corner,
                       std::string  dtype)
{
    const DType parsed_dtype = parse_dtype(dtype);
    switch (parsed_dtype) {
        case DType::Float32:
            return create_plane_typed<float>(nx,
                                             ny,
                                             plane,
                                             static_cast<float>(dx),
                                             with_cross_diagonal,
                                             low_corner);
        case DType::Float64:
            return create_plane_typed<double>(
                nx, ny, plane, dx, with_cross_diagonal, low_corner);
        default:
            throw std::invalid_argument(
                "create_plane() supports float32 and float64 coordinates.");
    }
}

}  // namespace

void register_geometry(py::module_& m)
{
    m.def("create_plane",
          &create_plane,
          py::arg("nx"),
          py::arg("ny"),
          py::arg("plane")               = 1,
          py::arg("dx")                  = 1.0,
          py::arg("with_cross_diagonal") = false,
          py::arg("low_corner")          = py::make_tuple(0.0, 0.0, 0.0),
          py::arg("dtype")               = "float32",
          "Create a regular RXMesh plane and return (vertices, faces) NumPy "
          "arrays.");
}

}  // namespace pyrxmesh_py
