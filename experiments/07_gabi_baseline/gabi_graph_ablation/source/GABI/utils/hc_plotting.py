import pyvista as pv
import numpy as np
import torch
from typing import Optional, Union
from torch_geometric.data import Data


def plot_car_mesh(graph: Data, scalar_name: str = 'u',
                  scalar_data: Optional[np.ndarray] = None,
                  cmap: str = 'coolwarm', camera_position: Optional[list] = None,
                  show_edges: bool = False, title: Optional[str] = None,
                  window_size: tuple = (800, 600)):
    """
    Visualize a car mesh with scalar data.

    Args:
        graph: PyTorch Geometric Data object with pos, face, and optionally x
        scalar_name: Name of the scalar field to display
        scalar_data: Optional scalar data array (uses graph.x if None)
        cmap: Colormap name ('coolwarm', 'viridis', 'plasma', etc.)
        camera_position: Camera position as [position, focal_point, view_up]
        show_edges: Whether to show mesh edges
        title: Plot title
        window_size: Window size as (width, height)
    """
    # Convert to numpy if needed
    vertices = graph.pos.cpu().numpy() if torch.is_tensor(graph.pos) else graph.pos
    faces = graph.face.cpu().numpy().T if torch.is_tensor(graph.face) else graph.face.T

    # Get scalar data
    if scalar_data is None:
        if hasattr(graph, 'x') and graph.x is not None:
            scalar_data = graph.x.cpu().numpy() if torch.is_tensor(graph.x) else graph.x
            if scalar_data.ndim > 1:
                scalar_data = scalar_data[:, 0]  # Take first channel
        else:
            scalar_data = None
    elif torch.is_tensor(scalar_data):
        scalar_data = scalar_data.cpu().numpy()

    # Flatten scalar data if needed
    if scalar_data is not None and scalar_data.ndim > 1:
        scalar_data = scalar_data.flatten()

    # Create PyVista mesh
    # PyVista expects faces as [n_vertices, v0, v1, v2, n_vertices, v0, v1, v2, ...]
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()
    mesh = pv.PolyData(vertices, faces_flat)

    # Add scalar data
    if scalar_data is not None:
        mesh.point_data[scalar_name] = scalar_data

    # Create plotter
    plotter = pv.Plotter(window_size=window_size)

    # Add mesh
    mesh_kwargs = {
        'cmap': cmap,
        'show_scalar_bar': True,
        'scalar_bar_args': {
            'title': scalar_name,
            'vertical': True,
            'position_x': 0.85,
            'position_y': 0.1,
            'width': 0.05,
            'height': 0.8
        }
    }

    if scalar_data is not None:
        mesh_kwargs['scalars'] = scalar_name

    if show_edges:
        mesh_kwargs['show_edges'] = True
        mesh_kwargs['edge_color'] = 'black'
        mesh_kwargs['line_width'] = 0.5

    plotter.add_mesh(mesh, **mesh_kwargs)

    # Set camera position
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        # Default: isometric view
        plotter.camera_position = [
            np.array([2, 3, 2]),  # position
            (0.0, 0.0, 0.0),      # focal point
            (0, 0, 1)              # view up
        ]

    # Show axes and display
    plotter.show_axes()
    if title:
        plotter.add_text(title, position='upper_edge', font_size=12)

    plotter.show()


def quick_pred_vs_true_plot(y_true: Union[np.ndarray, torch.Tensor],
                            y_pred: Union[np.ndarray, torch.Tensor],
                            graph: Data,
                            scalar_name: str = 'u',
                            cmap: str = 'coolwarm',
                            window_size: tuple = (1400, 500)):
    """
    Compare predicted vs true Helmholtz solutions side-by-side.

    Args:
        y_true: Ground truth solution (n_nodes,) or (n_nodes, 1)
        y_pred: Predicted solution (n_nodes,) or (n_nodes, 1)
        graph: PyTorch Geometric Data object with pos and face
        scalar_name: Name of the scalar field
        cmap: Colormap name
        window_size: Window size as (width, height)

    Returns:
        dict: Error statistics (mae, rmse, max_error)
    """
    # Convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().detach().numpy()

    # Flatten if needed
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    # Compute errors
    error = np.abs(y_pred - y_true)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error ** 2))
    max_error = np.max(error)

    stats = {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'rel_error': mae / (np.abs(y_true).mean() + 1e-10)
    }

    # Get mesh data
    vertices = graph.pos.cpu().numpy() if torch.is_tensor(graph.pos) else graph.pos
    faces = graph.face.cpu().numpy().T if torch.is_tensor(graph.face) else graph.face.T
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()

    # Create three meshes
    mesh_true = pv.PolyData(vertices, faces_flat)
    mesh_pred = pv.PolyData(vertices, faces_flat)
    mesh_error = pv.PolyData(vertices, faces_flat)

    mesh_true.point_data[scalar_name] = y_true
    mesh_pred.point_data[scalar_name] = y_pred
    mesh_error.point_data['Absolute Error'] = error

    # Shared color limits for true and pred
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    # Create plotter with 3 subplots
    plotter = pv.Plotter(shape=(1, 3), window_size=window_size, border=False)

    # Camera position for all subplots
    camera_pos = [np.array([2, 3, 2]), (0.0, 0.0, 0.0), (0, 0, 1)]

    # Subplot 1: Ground Truth
    plotter.subplot(0, 0)
    plotter.add_text(f'Ground Truth\n({scalar_name})', font_size=10, position='upper_edge')
    plotter.add_mesh(mesh_true, scalars=scalar_name, cmap=cmap, clim=[vmin, vmax])
    plotter.camera_position = camera_pos

    # Subplot 2: Prediction
    plotter.subplot(0, 1)
    plotter.add_text(f'Prediction\n(MAE={mae:.4f}, RMSE={rmse:.4f})',
                    font_size=10, position='upper_edge')
    plotter.add_mesh(mesh_pred, scalars=scalar_name, cmap=cmap, clim=[vmin, vmax])
    plotter.camera_position = camera_pos

    # Subplot 3: Error
    plotter.subplot(0, 2)
    plotter.add_text(f'Absolute Error\n(Max={max_error:.4f})',
                    font_size=10, position='upper_edge')
    plotter.add_mesh(mesh_error, scalars='Absolute Error', cmap='hot')
    plotter.camera_position = camera_pos

    # Link views for synchronized rotation
    plotter.link_views()

    plotter.show()

    return stats


def quick_data_plot(y: Union[np.ndarray, torch.Tensor],
                   graph: Data,
                   scalar_name: str = 'u',
                   cmap: str = 'coolwarm',
                   show_stats: bool = True,
                   camera_position: Optional[list] = None,
                   window_size: tuple = (800, 700)):
    """
    Quick visualization of scalar data on car mesh with statistics.

    Args:
        y: Scalar data array (n_nodes,) or (n_nodes, 1)
        graph: PyTorch Geometric Data object with pos and face
        scalar_name: Name of the scalar field
        cmap: Colormap name
        show_stats: Whether to show statistics in title
        camera_position: Camera position as [position, focal_point, view_up]
        window_size: Window size as (width, height)

    Returns:
        dict: Statistics of the scalar field
    """
    # Convert to numpy
    if torch.is_tensor(y):
        y = y.cpu().detach().numpy()

    # Flatten if needed
    if y.ndim > 1:
        y = y.flatten()

    # Compute statistics
    stats = {
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y),
        'median': np.median(y)
    }

    # Get mesh data
    vertices = graph.pos.cpu().numpy() if torch.is_tensor(graph.pos) else graph.pos
    faces = graph.face.cpu().numpy().T if torch.is_tensor(graph.face) else graph.face.T
    faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()

    # Create mesh
    mesh = pv.PolyData(vertices, faces_flat)
    mesh.point_data[scalar_name] = y

    # Create plotter
    plotter = pv.Plotter(window_size=window_size)

    # Create title with statistics
    if show_stats:
        title = (f"{scalar_name}\n"
                f"μ={stats['mean']:.4f}, σ={stats['std']:.4f}\n"
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    else:
        title = scalar_name

    plotter.add_text(title, position='upper_edge', font_size=12)

    # Add mesh
    plotter.add_mesh(mesh, scalars=scalar_name, cmap=cmap,
                    show_scalar_bar=True,
                    scalar_bar_args={
                        'title': scalar_name,
                        'vertical': True,
                        'position_x': 0.85,
                        'position_y': 0.1,
                        'width': 0.05,
                        'height': 0.8
                    })

    # Set camera position
    if camera_position is not None:
        plotter.camera_position = camera_position
    else:
        # Default: isometric view
        plotter.camera_position = [
            np.array([2, 3, 2]),
            (0.0, 0.0, 0.0),
            (0, 0, 1)
        ]

    plotter.show_axes()
    plotter.show()

    return stats


def plot_forcing_pattern(graph: Data, cmap: str = 'hot',
                        window_size: tuple = (800, 700)):
    """
    Visualize the forcing pattern (f) on the car mesh.

    Args:
        graph: PyTorch Geometric Data object with f attribute
        cmap: Colormap name
        window_size: Window size as (width, height)
    """
    if not hasattr(graph, 'f'):
        raise ValueError("Graph does not have forcing attribute 'f'")

    forcing = graph.f.cpu().numpy() if torch.is_tensor(graph.f) else graph.f

    return quick_data_plot(
        y=forcing,
        graph=graph,
        scalar_name='f',
        cmap=cmap,
        show_stats=True,
        window_size=window_size
    )


if __name__ == '__main__':
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from datatools import HelmholtzCarDataset

    parser = argparse.ArgumentParser(description='Plot Helmholtz car field from dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to Helmholtz car dataset (.pkl file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to plot')
    parser.add_argument('--field', type=str, default='solution',
                       choices=['solution', 'forcing'],
                       help='Field to plot (solution=u or forcing=f)')
    parser.add_argument('--show-edges', action='store_true', help='Show mesh edges')

    args = parser.parse_args()

    # Load dataset
    dataset = HelmholtzCarDataset(data_path=args.data)

    # Get sample
    if args.sample >= len(dataset):
        print(f"Error: Sample index {args.sample} out of range (dataset has {len(dataset)} samples)")
        sys.exit(1)

    data = dataset[args.sample]

    # Plot based on field type
    if args.field == 'forcing':
        plot_forcing_pattern(data, cmap='hot')
    else:  # solution
        # Extract solution from data.y (first channel if multi-dimensional)
        if hasattr(data, 'y') and data.y is not None:
            y = data.y
            if y.ndim > 1:
                y = y[:, 0]
            quick_data_plot(
                y=y,
                graph=data,
                scalar_name='u',
                cmap='coolwarm',
                show_stats=True
            )
        else:
            print("Error: Dataset does not have 'y' attribute for solution")
