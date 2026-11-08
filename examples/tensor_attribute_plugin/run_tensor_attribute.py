"""Tensor <-> Attribute interop

From this directory, with CUDA-enabled Torch and PyRXMesh already installed:
    cd examples\tensor_attribute_plugin
    python -m pip install -v --no-build-isolation .
    python run_tensor_attribute.py
"""

import torch

import pyrxmesh as rx
from rxmesh_tensor_attribute import transform_attribute, transform_tensor


def attr_tensor(mesh, channels):
    n = mesh.num_vertices
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream()
    stream_ptr = stream.cuda_stream

    # Tensor row i always means RXMesh LINEAR vertex i, not input OBJ vertex i.
    # Convert externally supplied input-order data once, then keep computations
    # in linear order. Gathering below copies; the inverse gather restores
    # order.
    input_order = torch.arange(n * channels, dtype=torch.float32, device=device)
    input_order = input_order.reshape(n, channels)
    linear_to_input = torch.as_tensor(
        mesh.linear_to_global(
            rx.ElementKind.Vertex),
        device=device,
        dtype=torch.long)
    input_to_linear = torch.as_tensor(
        mesh.global_to_linear(
            rx.ElementKind.Vertex),
        device=device,
        dtype=torch.long)
    values = input_order[linear_to_input]
    torch.testing.assert_close(values[input_to_linear], input_order)

    source = mesh.add_vertex_attribute(
        f"source_{channels}", dim=channels, dtype="float32",
        layout="soa", location="device",
    )
    result = mesh.add_vertex_attribute(
        f"result_{channels}", dim=channels, dtype="float32",
        layout="soa", location="device",
    )

    # Tensor -> Attribute: COPY into existing attribute-owned storage. Changes to
    # values afterward do not change source. The zero-copy alternative is
    # mesh.add_vertex_attribute_from_torch_view(), shown below.
    source.from_torch_copy(values, target="device")

    # Attribute -> Tensor: ZERO COPY. Shape is (n, channels),
    # strides are (1, n). For channels > 1 this is not Torch-contiguous.
    # The tensor retains the attribute owner. Do not release/reallocate its
    # underlying storage while a view is still in use.
    source_view = source.to_torch("device")
    result_view = result.to_torch("device")
    # In-place Torch edits immediately change the attribute.
    source_view.add_(10)
    torch.testing.assert_close(source_view, values + 10)
    independent = source_view.clone()  # COPY: an independent Torch allocation.

    # Attribute -> RXMesh kernel -> Attribute -> Torch, sharing the output
    # buffer.
    transform_attribute(mesh, source, result, stream=stream_ptr)
    torch.testing.assert_close(result_view, 2 * (values + 10) + 1)

    # Torch-owned input and output -> RXMesh kernel: ZERO COPY at the boundary.
    # The kernel uses context.linear_id(vh) and the actual tensor strides.
    tensor_output = torch.empty_like(values)
    transform_tensor(mesh, values, tensor_output, stream=stream_ptr)
    torch.testing.assert_close(tensor_output, 2 * values + 1)

    # Mixed ownership: direct tensor input, output written into attribute
    # storage.
    transform_tensor(mesh, values, result_view, stream=stream_ptr)
    torch.testing.assert_close(result_view, tensor_output)

    # Reverse mix: read an attribute's strided tensor view, write Torch storage.
    # This uses the same tensor kernel, with no contiguous() conversion or copy.
    transform_tensor(mesh, source_view, tensor_output, stream=stream_ptr)
    torch.testing.assert_close(tensor_output, 2 * independent + 1)

    # Write into an existing attribute through its view: this operation COPIES
    # values. The view itself remains an alias of result, not of tensor_output.
    result_view.copy_(tensor_output)

    # DLPack is the same underlying interchange used by
    # to_torch/from_torch_copy.
    dlpack_view = torch.from_dlpack(result)
    assert dlpack_view.data_ptr() == result_view.data_ptr()
    source.from_dlpack_copy(tensor_output, target="device")  # COPY again.
    torch.testing.assert_close(source_view, tensor_output)

    # Torch-owned Tensor -> real RXMesh Attribute: ZERO COPY. Allocate compatible
    # SoA strides directly, instead of converting an existing row-major tensor.
    # A normal (N, 1) contiguous tensor works too. For channels > 1, a normal
    # contiguous (N, channels) tensor is rejected; the factory never silently
    # rearranges values or changes their linear element order.
    shared = torch.empty_strided(
        (n, channels), (1, n), dtype=torch.float32, device=device
    )
    shared.copy_(values)  # This initialization copies; wrapping below does not.
    borrowed = mesh.add_vertex_attribute_from_torch_view(
        f"borrowed_{channels}", shared)
    assert borrowed.to_torch("device").data_ptr() == shared.data_ptr()
    # Keep the tensor's storage fixed: do not resize it while the attribute is
    # attached. Native writes share its values but do not build an autograd graph.
    # Use the ordinary ATTRIBUTE kernel to write back into Torch-owned storage.
    transform_attribute(mesh, source, borrowed, stream=stream_ptr)
    torch.testing.assert_close(shared, 2 * source_view + 1)
    shared.add_(5)
    torch.testing.assert_close(borrowed.to_torch("device"), shared)
    # Ownership is retained by the attribute and its mesh registration. Deleting
    # the original tensor variable does not invalidate the borrowed attribute.
    stream.synchronize()
    del shared
    transform_attribute(mesh, borrowed, result, stream=stream_ptr)
    torch.testing.assert_close(result_view, 2 * (2 * source_view + 6) + 1)
    # Edge/face data use the corresponding add_edge/face_attribute_from_torch_view
    # factories. All infer dtype and channel count and require CUDA SoA storage.

    # Complete all work before the local tensor owners leave scope. For custom
    # streams, also establish dependencies before reading on a different stream.
    # Keep the mesh and attribute owners alive along with their borrowed views.
    stream.synchronize()
    print(f"{channels} value(s)/vertex: shared views and copy paths passed")


def main():
    torch.cuda.set_device(0)
    rx.init(0)
    vertices, faces = rx.create_plane(32, 32)

    mesh = rx.RXMeshStatic(vertices, faces, patch_size=256)
    for channels in (1, 3):
        attr_tensor(mesh, channels)

    # Optional HOST path: copying between CPU tensors and host attribute storage
    # uses the same API. A CPU tensor view aliases only the HOST allocation;
    # host/device mirrors are separate allocations and require explicit move().
    host = mesh.add_vertex_attribute("host_values", dim=1, location="host")
    host.from_torch_copy(torch.ones(mesh.num_vertices, 1), target="host")
    # ZERO COPY of host memory, not a GPU view.
    host_view = host.to_torch("host")
    host_view.mul_(3)
    host.move("host", "device")  # COPY to device, making the CPU edits visible.
    torch.testing.assert_close(host.to_torch("device"), host_view.cuda())
    # After GPU edits, host.move("device", "host") refreshes the host
    # allocation.

    # AoS/AoSoA attributes support explicit copies and handle-based kernels, but
    # cannot be exported as zero-copy tensors. Use SoA when sharing with Torch.
    # A view is mutable and will see future kernel writes. Use clone() when an
    # independent result is needed. contiguous() may also copy; it is unnecessary
    # for this example's stride-aware tensor kernel.


if __name__ == "__main__":
    main()
