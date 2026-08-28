from __future__ import annotations

import argparse
from pathlib import Path

import torch

import pyrxmesh as rx
import rxmesh_diff_energy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input OBJ mesh")
    parser.add_argument("--w0", type=float, default=0.5)
    parser.add_argument("--w1", type=float, default=0.25)
    args = parser.parse_args()

    rx.init(0)
    mesh_path = Path(args.input)
    mesh = rx.RXMeshStatic(str(mesh_path), patch_size=256)
    energy = rxmesh_diff_energy.make_energy(
        mesh, {"w0": args.w0, "w1": args.w1}
    )

    # mesh.vertices() already uses the linear row order required by the energy.
    x = torch.as_tensor(
        mesh.vertices(), dtype=torch.float32, device="cuda"
    ).requires_grad_()
    loss = energy.torch(x)
    loss.backward()

    print(f"mesh: {mesh_path}")
    print(f"vertices: {mesh.num_vertices}, edges: {mesh.num_edges}")
    print(f"terms: {energy.term_count}")
    print(f"loss: {loss.item():.8g}")
    print(
        f"gradient shape: {tuple(x.grad.shape)}, "
        f"stride: {tuple(x.grad.stride())}"
    )


if __name__ == "__main__":
    main()
