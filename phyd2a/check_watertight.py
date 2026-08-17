import os
import numpy as np
import trimesh
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--obj_dir", type=str, required=True, help="The obj directory to check for watertightness.")
args = parser.parse_args()
obj_dir = args.obj_dir

for f in os.listdir(obj_dir):
    if not f.endswith(".obj"):
        continue

    path = os.path.join(obj_dir, f)
    mesh = trimesh.load_mesh(path, process=False, maintain_order=True)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

    boundary_edges = unique_edges[counts == 1]
    nonmanifold_edges = unique_edges[counts > 2]

    print("\n", f)
    print("watertight:", mesh.is_watertight)
    print("vertices:", len(mesh.vertices), "faces:", len(mesh.faces))
    print("boundary edges:", len(boundary_edges))
    print("non-manifold edges:", len(nonmanifold_edges))
    print("euler number:", mesh.euler_number)
