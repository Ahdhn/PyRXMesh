from __future__ import annotations

import argparse

import pyrxmesh as rxmesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and visualize an RXMesh OBJ file.")
    parser.add_argument("--input", required=True, help="Path to an OBJ mesh file.")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id.")
    args = parser.parse_args()

    rxmesh.init(args.device_id)
    mesh = rxmesh.RXMeshStatic(args.input)

    print("RXMeshStatic(")
    print(f"  vertices={mesh.num_vertices},")
    print(f"  edges={mesh.num_edges},")
    print(f"  faces={mesh.num_faces},")
    print(f"  patches={mesh.num_patches},")
    print(f"  components={mesh.num_components},")
    print(f"  closed={mesh.is_closed()},")
    print(f"  edge_manifold={mesh.is_edge_manifold()}")
    print(")")

    rxmesh.show()


if __name__ == "__main__":
    main()