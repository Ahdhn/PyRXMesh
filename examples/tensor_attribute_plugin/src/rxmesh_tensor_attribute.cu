#include <pybind11/pybind11.h>

#include "pyrxmesh/plugin_api.h"

namespace py = pybind11;
using namespace rxmesh;

// This example uses Python's Tensor metadata and data_ptr()
// This small, device-safe struct is captured
// by the lambda. Never capture a py::object in GPU code.
struct TensorView
{
    float*  data;
    int64_t channels;
    int64_t row_stride;
    int64_t column_stride;

    __device__ float& operator()(uint32_t row, int64_t column) const
    {
        return data[row * row_stride + column * column_stride];
    }
};

TensorView tensor_view(py::handle tensor, size_t rows, bool output)
{
    const auto torch = py::module_::import("torch");
    if (!py::isinstance(tensor, torch.attr("Tensor")) ||
        !tensor.attr("is_cuda").cast<bool>() ||
        !tensor.attr("dtype").is(torch.attr("float32")) ||
        !tensor.attr("layout").is(torch.attr("strided"))) {
        throw py::value_error("Expected a strided CUDA float32 torch.Tensor.");
    }
    int device = 0;
    CUDA_ERROR(cudaGetDevice(&device));
    if (tensor.attr("get_device")().cast<int>() != device) {
        throw py::value_error("Tensor must be on RXMesh's current CUDA device.");
    }
    if (tensor.attr("dim")().cast<int>() != 2 ||
        tensor.attr("size")(0).cast<size_t>() != rows ||
        tensor.attr("size")(1).cast<int64_t>() <= 0) {
        throw py::value_error("Expected shape (mesh.num_vertices, channels).");
    }
    // Raw writes are not tracked by Torch autograd, including its version counter.
    // Reject grad-tracked and lazy-negative tensors instead of silently using
    // their storage with different semantics.
    if (tensor.attr("requires_grad").cast<bool>() ||
        tensor.attr("is_neg")().cast<bool>()) {
        throw py::value_error("Use ordinary tensors without requires_grad or lazy negation.");
    }
    TensorView view{
        reinterpret_cast<float*>(tensor.attr("data_ptr")().cast<uintptr_t>()),
        tensor.attr("size")(1).cast<int64_t>(),
        tensor.attr("stride")(0).cast<int64_t>(),
        tensor.attr("stride")(1).cast<int64_t>(),
    };
    // Read input with its actual strides. For output, accept the two compact
    // layouts used here and disallow overlapping/expanded output storage.
    const bool row_major = view.column_stride == 1 && view.row_stride == view.channels;
    const bool soa = view.row_stride == 1 && view.column_stride == int64_t(rows);
    if (view.row_stride < 0 || view.column_stride < 0 ||
        (output && !row_major && !soa)) {
        throw py::value_error("Output must be compact row-major or SoA; input strides must be nonnegative.");
    }
    return view;
}

// Attribute path: RXMesh's wrapper maps each vertex handle to its storage.
// Both attributes must belong to this mesh. SoA, AoS, and AoSoA can all be
// accessed by handle; only SoA supports the public zero-copy Torch export.
void transform_attribute(py::object mesh_object,
                         py::object input_object,
                         py::object output_object,
                         float scale, float bias, uintptr_t stream)
{
    auto input = pyrxmesh::vertex_attribute<float>(input_object);
    auto output = pyrxmesh::vertex_attribute<float>(output_object);
    const auto rows = pyrxmesh::mesh(mesh_object).get_num_vertices();
    if (!input.is_device_allocated() || !output.is_device_allocated() ||
        input.rows() != rows || output.rows() != rows ||
        input.get_num_attributes() != output.get_num_attributes()) {
        throw py::value_error("Expected matching device vertex attributes for this mesh.");
    }
    pyrxmesh::for_each_vertex(
        mesh_object,
        [input, output, scale, bias] __device__(const VertexHandle& vh) mutable {
            for (int c = 0; c < input.get_num_attributes(); ++c) {
                output(vh, c) = scale * input(vh, c) + bias;
            }
        }, reinterpret_cast<cudaStream_t>(stream));
}

// Tensor path: borrowing pointers does NOT create an RXMesh Attribute, allocate
// storage, or copy values. The tensors can be Torch-owned or attr.to_torch()
// views. Input/output may be identical, but must not otherwise overlap.
void transform_tensor(py::object mesh_object,
                      py::object input_object,
                      py::object output_object,
                      float scale, float bias, uintptr_t stream)
{
    const auto rows = pyrxmesh::mesh(mesh_object).get_num_vertices();
    const auto input = tensor_view(input_object, rows, false);
    const auto output = tensor_view(output_object, rows, true);
    if (input.channels != output.channels) {
        throw py::value_error("Input and output must have the same channel count.");
    }
    const auto context = pyrxmesh::context(mesh_object);
    pyrxmesh::for_each_vertex(
        mesh_object,
        [context, input, output, scale, bias] __device__(const VertexHandle& vh) {
            // Handle.local_id() is local to a patch; it is NOT a tensor row.
            const uint32_t row = context.linear_id(vh);
            for (int64_t c = 0; c < input.channels; ++c) {
                output(row, c) = scale * input(row, c) + bias;
            }
        }, reinterpret_cast<cudaStream_t>(stream));
    // Launches are asynchronous. The caller keeps tensors/attributes/mesh alive
    // until this stream completes. Running on Torch's current stream orders the
    // kernel with Torch operations; record_stream() alone does not retain owners.
}

PYBIND11_MODULE(_rxmesh_tensor_attribute, m)
{
    pyrxmesh::require_compatible_runtime(m);
    m.def("transform_attribute", &transform_attribute,
          py::arg("mesh"), py::arg("input"), py::arg("output"),
          py::arg("scale") = 2.0f, py::arg("bias") = 1.0f,
          py::arg("stream") = uintptr_t(0));
    m.def("transform_tensor", &transform_tensor,
          py::arg("mesh"), py::arg("input"), py::arg("output"),
          py::arg("scale") = 2.0f, py::arg("bias") = 1.0f,
          py::arg("stream") = uintptr_t(0));
}
