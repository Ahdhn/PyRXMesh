from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SHADOW_PATHS = {THIS_DIR, REPO_ROOT}


def _is_shadow_path(path: str) -> bool:
    resolved = Path(path).resolve() if path else Path.cwd().resolve()
    return resolved in SHADOW_PATHS


sys.path = [
    path
    for path in sys.path
    if not _is_shadow_path(path)
]

import pyrxmesh as rx  # noqa: E402
import rxmesh_edge_lengths  # noqa: E402


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
