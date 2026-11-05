"""
Generate Helmholtz Car Dataset

Loads car meshes (PLY format), solves the Helmholtz equation with random forcing,
and saves as PyTorch Geometric graphs for training.

The Helmholtz equation solved is:
    (Δ + k²) u = f
where k is the wavenumber, and f is a localized Gaussian forcing.
"""

import os
import argparse
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import trimesh
import scipy.sparse
from scipy.sparse.linalg import LinearOperator, lgmres
from tqdm import tqdm


def load_car_mesh(ply_path):
    """Load a car mesh from PLY file"""
    mesh = trimesh.load(ply_path, process=False)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Convert faces to edge indices
    edge_index = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ], dim=0).t().contiguous()

    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    return Data(pos=vertices, edge_index=edge_index, face=faces.t().contiguous())


def preprocess_mesh(graph):
    """
    Preprocess mesh:
    - Normalize to unit cube
    - Check watertightness
    - Compute edge lengths
    """
    vertices = graph.pos.cpu().numpy()
    faces = graph.face.cpu().numpy().T

    # Build trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Normalize to unit cube
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    max_dim = np.max(size)

    if max_dim == 0:
        raise ValueError("Mesh has zero-size bounding box")

    mesh.vertices = (mesh.vertices - bounds[0]) / max_dim

    # Check mesh validity
    if len(mesh.split(only_watertight=False)) > 1:
        return None  # Skip disconnected components

    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight")
        return None

    # Build new graph with normalized vertices
    new_vertices = torch.from_numpy(mesh.vertices).float()
    new_faces = torch.from_numpy(mesh.faces.T).long()

    # Compute edges and edge lengths
    edges = np.concatenate([
        mesh.faces[:, [0, 1]],
        mesh.faces[:, [1, 2]],
        mesh.faces[:, [2, 0]]
    ], axis=0)

    edges_sorted = np.sort(edges, axis=1)
    unique_edges = np.unique(edges_sorted, axis=0)

    # Compute edge lengths
    v = mesh.vertices
    edge_lengths = np.linalg.norm(v[unique_edges[:, 0]] - v[unique_edges[:, 1]], axis=1)

    new_edge_index = torch.from_numpy(unique_edges.T).long()
    new_edge_attr = torch.from_numpy(edge_lengths).float()

    return Data(
        pos=new_vertices,
        face=new_faces,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr
    )


def cotangent_laplacian(vertices, faces):
    """Build cotangent Laplacian matrix"""
    n_vertices = vertices.shape[0]
    I, J, W = [], [], []

    for tri in faces:
        pts = vertices[tri]
        for i in range(3):
            i1, i2, i3 = tri[i], tri[(i+1)%3], tri[(i+2)%3]
            v1 = pts[(i+1)%3] - pts[i]
            v2 = pts[(i+2)%3] - pts[i]

            cross = np.cross(v1, v2)
            cross_norm = np.linalg.norm(cross)

            if cross_norm > 1e-10:
                cot_angle = np.dot(v1, v2) / cross_norm
                I += [i2, i3]
                J += [i3, i2]
                W += [cot_angle/2, cot_angle/2]

    L = scipy.sparse.coo_matrix((W, (I, J)), shape=(n_vertices, n_vertices))
    L = L + L.T
    diag = np.array(L.sum(axis=1)).flatten()
    L = scipy.sparse.diags(diag) - L

    return L


def mass_matrix(vertices, faces):
    """Build lumped mass matrix"""
    n_vertices = vertices.shape[0]
    M = np.zeros(n_vertices)

    for tri in faces:
        pts = vertices[tri]
        area = np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0])) / 2
        for idx in tri:
            M[idx] += area / 3

    return scipy.sparse.diags(M)


def solve_helmholtz(graph, k_squared=500, damping_factor=0.2, seed=None):
    """
    Solve Helmholtz equation on the mesh with random Gaussian forcing.

    Args:
        graph: PyTorch Geometric Data object with pos and face
        k_squared: Wavenumber squared (controls frequency)
        damping_factor: Damping coefficient (controls dissipation)
        seed: Random seed for forcing location

    Returns:
        dict with solution u, forcing f, and metadata
    """
    V = graph.pos.cpu().numpy()
    F = graph.face.cpu().numpy().T

    # Build operators
    L = cotangent_laplacian(V, F).astype(np.complex128)
    M = mass_matrix(V, F).astype(np.complex128)

    # Random Gaussian forcing (localized near bottom of mesh)
    if seed is not None:
        np.random.seed(seed)

    # Choose forcing center from lower part of mesh
    center_candidates = np.where(V[:, 2] < np.percentile(V[:, 2], 20))[0]
    if len(center_candidates) == 0:
        center_candidates = np.arange(V.shape[0])

    center_idx = np.random.choice(center_candidates)
    f = np.exp(-((V - V[center_idx])**2).sum(axis=1) / (0.1**2)) * 0.1
    f = f.astype(np.complex128)

    # Build complex system: A = L + (k² + ic) * M
    c = damping_factor * k_squared
    complex_k_squared = -k_squared + 1j * c
    A = L + complex_k_squared * M

    A_linop = LinearOperator(
        matvec=lambda x: A @ x,
        dtype=np.complex128,
        shape=A.shape
    )

    # Solve using LGMRES
    u, info = lgmres(A_linop, f, rtol=1e-5, maxiter=10000)

    if info != 0:
        print(f"Warning: LGMRES did not converge (info={info})")
        return None

    return {
        'u': u,
        'f': f,
        'center_idx': center_idx,
        'k_squared': k_squared,
        'damping': c
    }


def generate_helmholtz_dataset(
    mesh_dir,
    output_path,
    n_samples=1000,
    k_squared=500,
    damping_factor=0.2
):
    """
    Generate Helmholtz car dataset.

    Args:
        mesh_dir: Directory containing PLY mesh files
        output_path: Path to save pickle file
        n_samples: Number of samples to generate
        k_squared: Wavenumber squared
        damping_factor: Damping coefficient (as fraction of k_squared)
    """
    # Load all mesh files
    mesh_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.ply')])

    if len(mesh_files) == 0:
        raise ValueError(f"No PLY files found in {mesh_dir}")

    print(f"Found {len(mesh_files)} mesh files")
    print(f"Generating {n_samples} samples...")

    all_graphs = []

    for i in tqdm(range(n_samples)):
        # Randomly select a mesh
        mesh_idx = np.random.randint(0, len(mesh_files))
        mesh_path = os.path.join(mesh_dir, mesh_files[mesh_idx])

        # Load and preprocess
        graph = load_car_mesh(mesh_path)
        graph = preprocess_mesh(graph)

        if graph is None:
            print(f"Skipping sample {i} (invalid mesh)")
            continue

        # Solve Helmholtz equation
        result = solve_helmholtz(graph, k_squared=k_squared,
                                damping_factor=damping_factor, seed=i)

        if result is None:
            print(f"Skipping sample {i} (solver failed)")
            continue

        # Store solution in graph
        graph.x = torch.from_numpy(np.abs(result['u'])).float().unsqueeze(-1)  # Magnitude
        graph.f = torch.from_numpy(np.abs(result['f'])).float()  # Forcing
        graph.k_squared = torch.tensor([result['k_squared']], dtype=torch.float)
        graph.damping = torch.tensor([result['damping']], dtype=torch.float)
        graph.center_idx = torch.tensor([result['center_idx']], dtype=torch.long)

        all_graphs.append(graph)

    print(f"Generated {len(all_graphs)} valid samples")

    # Save to pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_graphs, f)

    print(f"Saved dataset to {output_path}")

    return all_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Helmholtz car dataset")
    parser.add_argument('--mesh_dir', type=str, required=True,
                       help='Directory containing PLY mesh files')
    parser.add_argument('--output', type=str, default='data/helmholtz_car.pkl',
                       help='Output pickle file path')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--k_squared', type=float, default=500,
                       help='Wavenumber squared')
    parser.add_argument('--damping', type=float, default=0.2,
                       help='Damping factor (as fraction of k_squared)')

    args = parser.parse_args()

    generate_helmholtz_dataset(
        mesh_dir=args.mesh_dir,
        output_path=args.output,
        n_samples=args.n_samples,
        k_squared=args.k_squared,
        damping_factor=args.damping
    )
