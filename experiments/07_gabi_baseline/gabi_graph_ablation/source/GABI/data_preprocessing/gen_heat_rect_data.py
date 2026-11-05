"""
Generate Heat Rectangle Dataset

Creates 2D unstructured triangular meshes on rectangles with varying dimensions
and boundary conditions, then solves the steady-state heat equation.

The heat equation solved is:
    -∇·(k∇u) = f
with Dirichlet boundary conditions on all edges.
"""

import argparse
import pickle
import os
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def create_triangular_mesh(width, height, resolution=10):
    """
    Create a 2D triangular mesh on a rectangle.

    Args:
        width: Rectangle width
        height: Rectangle height
        resolution: Mesh resolution (approximate number of nodes along shorter dimension)

    Returns:
        points: Node positions (N, 2)
        triangles: Triangle connectivity (M, 3)
        tri: Delaunay triangulation object
    """
    # Determine grid size based on aspect ratio
    if width > height:
        nx, ny = int(width/height * resolution), int(resolution)
    else:
        nx, ny = int(resolution), int(height/width * resolution)

    # Create regular grid
    x, y = np.meshgrid(np.linspace(0, width, nx), np.linspace(0, height, ny))

    # Add random perturbation to interior nodes
    x[:, 1:-1] += (width / (2 * nx)) * np.random.rand(ny, nx-2)
    y[1:-1, :] += (height / (2 * ny)) * np.random.rand(ny-2, nx)

    # Flatten to points
    points = np.column_stack([x.ravel(), y.ravel()])

    # Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices

    return points, triangles, tri


def local_stiffness_matrix(triangle_coords, conductivity=1.0):
    """
    Compute local stiffness matrix for a triangle element.

    Args:
        triangle_coords: (3, 2) array of triangle vertex coordinates
        conductivity: Thermal conductivity (assumed constant)

    Returns:
        ke: (3, 3) local stiffness matrix
        area: Triangle area
    """
    x = triangle_coords[:, 0]
    y = triangle_coords[:, 1]

    # Triangle area
    area = 0.5 * np.linalg.det(np.array([
        [1, x[0], y[0]],
        [1, x[1], y[1]],
        [1, x[2], y[2]]
    ]))

    if area <= 0:
        raise ValueError("Non-positive triangle area")

    # Gradients of linear shape functions
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])

    grad = np.vstack((b, c)) / (2.0 * area)

    # Local stiffness: ke[i,j] = k * area * (∇φ_i · ∇φ_j)
    ke = conductivity * area * grad.T @ grad

    return ke, area


def solve_heat_equation(width, height, boundary_conditions, resolution=10):
    """
    Solve steady-state heat equation on rectangle.

    Args:
        width: Rectangle width
        height: Rectangle height
        boundary_conditions: Tuple (u_left, u_right, u_top, u_bottom)
        resolution: Mesh resolution

    Returns:
        u: Solution field
        points: Node positions
        triangles: Triangle connectivity
        tri: Delaunay object
    """
    u_left, u_right, u_top, u_bottom = boundary_conditions

    # Create mesh
    points, triangles, tri = create_triangular_mesh(width, height, resolution)
    num_nodes = len(points)

    # Assemble stiffness matrix and load vector
    K = lil_matrix((num_nodes, num_nodes))
    F = np.zeros(num_nodes)

    for tri_nodes in triangles:
        coords = points[tri_nodes]
        ke, area = local_stiffness_matrix(coords)

        # Assemble into global matrix
        for i in range(3):
            for j in range(3):
                K[tri_nodes[i], tri_nodes[j]] += ke[i, j]

            # Load vector (assuming zero source term)
            F[tri_nodes[i]] += 0.0

    # Apply Dirichlet boundary conditions
    def get_boundary_value(x, y):
        """Get boundary value for a point"""
        if np.isclose(y, 0, atol=1e-6):
            return u_bottom
        elif np.isclose(y, height, atol=1e-6):
            return u_top
        elif np.isclose(x, 0, atol=1e-6):
            return u_left
        elif np.isclose(x, width, atol=1e-6):
            return u_right
        return None

    # Find boundary nodes
    boundary_nodes = np.unique(tri.convex_hull)

    # Initialize solution
    u = np.zeros(num_nodes)
    for i in boundary_nodes:
        bc_val = get_boundary_value(*points[i])
        if bc_val is not None:
            u[i] = bc_val

    # Convert to CSR format
    K = K.tocsr()

    # Adjust RHS for known boundary values
    F -= K @ u

    # Strongly enforce boundary conditions
    for i in boundary_nodes:
        K[i, :] = 0
        K[:, i] = 0
        K[i, i] = 1
        F[i] = u[i]

    # Solve linear system
    u = spsolve(K, F)

    return u, points, triangles, tri


def edges_from_triangulation(tri):
    """Extract unique edges and their lengths from Delaunay triangulation"""
    edges_list = []

    for triangle in tri.simplices:
        for e1, e2 in [[0, 1], [1, 2], [2, 0]]:
            edge = [triangle[e1], triangle[e2]]
            edges_list.append(tuple(sorted(edge)))

    # Remove duplicates
    unique_edges = np.array(list(set(edges_list)))

    # Compute edge lengths
    points = tri.points
    edge_lengths = np.linalg.norm(
        points[unique_edges[:, 0]] - points[unique_edges[:, 1]],
        axis=1
    )

    return unique_edges, edge_lengths


def generate_single_sample(width_range=(0.5, 2.0), height_range=(0.5, 2.0),
                           resolution=10):
    """
    Generate a single heat rectangle sample with random dimensions and BCs.

    Args:
        width_range: (min_width, max_width)
        height_range: (min_height, max_height)
        resolution: Mesh resolution

    Returns:
        PyTorch Geometric Data object
    """
    # Random dimensions
    width = np.random.uniform(*width_range)
    height = np.random.uniform(*height_range)

    # Random boundary conditions
    u_left = np.random.uniform(0, 1)
    u_right = np.random.uniform(0, 1)
    u_top = np.random.uniform(0, 1)
    u_bottom = np.random.uniform(0, 1)
    boundary_conditions = (u_left, u_right, u_top, u_bottom)

    # Solve heat equation
    u, points, triangles, tri = solve_heat_equation(
        width, height, boundary_conditions, resolution
    )

    # Extract edges
    edges, edge_lengths = edges_from_triangulation(tri)

    # Create PyTorch Geometric Data
    data = Data(
        pos=torch.tensor(points, dtype=torch.float),
        edge_index=torch.tensor(edges.T, dtype=torch.long),
        edge_attr=torch.tensor(edge_lengths, dtype=torch.float),
        face=torch.tensor(triangles.T, dtype=torch.long),
        x=torch.tensor(u, dtype=torch.float).reshape(-1, 1),
        boundary_conditions=torch.tensor(boundary_conditions, dtype=torch.float),
        dimensions=torch.tensor([width, height], dtype=torch.float)
    )

    return data


def generate_heat_rect_dataset(
    output_path,
    n_samples=1000,
    width_range=(0.5, 2.0),
    height_range=(0.5, 2.0),
    resolution=10
):
    """
    Generate complete heat rectangle dataset.

    Args:
        output_path: Path to save pickle file
        n_samples: Number of samples to generate
        width_range: (min_width, max_width) for random sampling
        height_range: (min_height, max_height) for random sampling
        resolution: Mesh resolution

    Returns:
        List of PyTorch Geometric Data objects
    """
    print(f"Generating {n_samples} heat rectangle samples...")
    print(f"  Width range: {width_range}")
    print(f"  Height range: {height_range}")
    print(f"  Resolution: {resolution}")

    samples = []
    for i in tqdm(range(n_samples)):
        try:
            data = generate_single_sample(width_range, height_range, resolution)
            samples.append(data)
        except Exception as e:
            print(f"Warning: Failed to generate sample {i}: {e}")
            continue

    print(f"Successfully generated {len(samples)} samples")

    # Save to pickle
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"Saved dataset to {output_path}")

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heat rectangle dataset")
    parser.add_argument('--output', type=str, default='data/heat_rect.pkl',
                       help='Output pickle file path')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--width_min', type=float, default=0.5,
                       help='Minimum rectangle width')
    parser.add_argument('--width_max', type=float, default=2.0,
                       help='Maximum rectangle width')
    parser.add_argument('--height_min', type=float, default=0.5,
                       help='Minimum rectangle height')
    parser.add_argument('--height_max', type=float, default=2.0,
                       help='Maximum rectangle height')
    parser.add_argument('--resolution', type=int, default=10,
                       help='Mesh resolution')

    args = parser.parse_args()

    generate_heat_rect_dataset(
        output_path=args.output,
        n_samples=args.n_samples,
        width_range=(args.width_min, args.width_max),
        height_range=(args.height_min, args.height_max),
        resolution=args.resolution
    )
