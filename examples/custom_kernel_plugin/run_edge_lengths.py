from __future__ import annotations

import argparse
from pathlib import Path

import pyrxmesh as rx
import rxmesh_edge_lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the custom PyRXMesh edge-length CUDA plugin."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input OBJ mesh")
    parser.add_argument("--device-id", default=0, type=int, help="CUDA device ID")
    args = parser.parse_args()

    rx.init(args.device_id)

    mesh = rx.RXMeshStatic(str(args.input))
    coords = mesh.input_vertex_coordinates()
    edge_lengths = mesh.add_edge_attribute(
        "edge_lengths", dtype="float32", dim=1
    )

    rxmesh_edge_lengths.compute_edge_lengths(mesh, coords, edge_lengths)
    values = edge_lengths.to_numpy()

    print(f"vertices: {mesh.num_vertices}")
    print(f"edges: {mesh.num_edges}")
    print(f"faces: {mesh.num_faces}")
    print(
        "edge length min/mean/max: "
        f"{values.min():.6f} / {values.mean():.6f} / {values.max():.6f}"
    )


if __name__ == "__main__":
    main()
