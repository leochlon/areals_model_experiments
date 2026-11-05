"""
Downsample Airfoil Dataset

Pre-process airfoil mesh data by downsampling to reduce computational cost.
The downsampled dataset is saved as a new zip file for efficient training.
"""

import torch
from zipfile import ZipFile
import io
import argparse
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm


def remove_close_points(points, features, node_type, tolerance):
    """
    Remove nodes that are too close to each other to thin the mesh.
    Preserves airfoil nodes (node_type == 1) while thinning fluid nodes.

    Args:
        points: Node positions (N, 2)
        features: Node features (N, F)
        node_type: Node type indicator (N,)
        tolerance: Minimum distance between points

    Returns:
        Mask of nodes to keep
    """
    tree = cKDTree(points)
    to_keep = np.ones(len(points), dtype=bool)
    is_airfoil = (node_type == 1.0)

    # Always keep airfoil nodes
    for i in range(len(points)):
        if not to_keep[i] or is_airfoil[i]:
            continue

        # Find nearby points
        indices = tree.query_ball_point(points[i], r=tolerance)

        # Remove this node from candidates
        if i in indices:
            indices.remove(i)

        # Mark nearby fluid nodes for removal
        for idx in indices:
            if not is_airfoil[idx]:
                to_keep[idx] = False

    return to_keep


def constrained_delaunay(points, node_type):
    """
    Perform constrained Delaunay triangulation avoiding all-airfoil triangles.

    Args:
        points: Node positions (N, 2)
        node_type: Node type indicator (N,)

    Returns:
        edges: Edge connectivity (E, 2)
        edge_attr: Edge attributes [dx, dy, length, face_surface] (E, 4)
    """
    is_airfoil = (node_type == 1.0)

    # Delaunay triangulation
    tri = Delaunay(points)
    all_faces = tri.simplices

    # Filter out triangles with 3 airfoil nodes (invalid for fluid mesh)
    valid_faces = []
    for face in all_faces:
        if np.sum(is_airfoil[face]) < 3:
            valid_faces.append(face)

    valid_faces = np.array(valid_faces)

    # Extract edges from valid faces
    edges_set = set()
    for face in valid_faces:
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            edges_set.add(edge)

    edges = np.array(list(edges_set))

    # Compute edge attributes
    edge_vectors = points[edges[:, 1]] - points[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    # Compute face surface for each edge (approximate as triangle area / 3)
    face_surfaces = np.zeros(len(edges))
    edge_to_idx = {tuple(sorted(edge)): i for i, edge in enumerate(edges)}

    for face in valid_faces:
        # Compute triangle area
        p0, p1, p2 = points[face]
        area = 0.5 * np.abs(np.cross(p1 - p0, p2 - p0))

        # Distribute area to edges
        for i in range(3):
            edge = tuple(sorted([face[i], face[(i+1)%3]]))
            if edge in edge_to_idx:
                face_surfaces[edge_to_idx[edge]] += area / 3

    # Combine edge attributes [dx, dy, length, face_surface]
    edge_attr = np.column_stack([
        edge_vectors[:, 0],
        edge_vectors[:, 1],
        edge_lengths,
        face_surfaces
    ])

    return edges, edge_attr


def downsample_mesh(data, tolerance):
    """
    Apply mesh thinning while preserving airfoil resolution.

    Args:
        data: PyTorch Geometric Data object
        tolerance: Distance tolerance for thinning

    Returns:
        Modified data object with downsampled mesh
    """
    # Convert to numpy
    pos = data.pos.cpu().numpy()
    x = data.x.cpu().numpy()
    node_type = data.node_type.cpu().numpy()

    # Remove close points (thin the mesh)
    keep_mask = remove_close_points(pos, x, node_type, tolerance)

    # Keep only selected nodes
    new_pos = pos[keep_mask]
    new_x = x[keep_mask]
    new_node_type = node_type[keep_mask]

    # Rebuild graph with constrained Delaunay
    new_edges, new_edge_attr = constrained_delaunay(new_pos, new_node_type)

    # Convert back to torch
    data.pos = torch.from_numpy(new_pos).float()
    data.x = torch.from_numpy(new_x).float()
    data.node_type = torch.from_numpy(new_node_type).long()
    data.edge_index = torch.from_numpy(new_edges.T).long()
    data.edge_attr = torch.from_numpy(new_edge_attr).float()

    return data


def main():
    parser = argparse.ArgumentParser(description='Downsample Airfoil Dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input zip file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output zip file')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                        help='Distance tolerance for downsampling (default: 1e-2)')
    args = parser.parse_args()

    print(f"Downsampling airfoil dataset: {args.input}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Output: {args.output}")

    # Read input zip
    with ZipFile(args.input, 'r') as zf_in:
        file_list = zf_in.namelist()
        print(f"Found {len(file_list)} samples")

        # Create output zip
        with ZipFile(args.output, 'w') as zf_out:
            for filename in tqdm(file_list, desc="Downsampling"):
                # Load data
                with zf_in.open(filename) as item:
                    stream = io.BytesIO(item.read())
                    data = torch.load(stream, weights_only=False)

                # Downsample mesh
                data = downsample_mesh(data, tolerance=args.tolerance)

                # Save to output zip
                buffer = io.BytesIO()
                torch.save(data, buffer)
                zf_out.writestr(filename, buffer.getvalue())

    print(f"Downsampled dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
