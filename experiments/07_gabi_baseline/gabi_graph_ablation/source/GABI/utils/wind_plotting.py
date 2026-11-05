import pyvista as pv
import numpy as np
import torch
from typing import Optional
from torch_geometric.data import Data


def create_terrain_voxels_pv(terrain_mask: np.ndarray, ds, **kwargs):
    """
    Create a voxel representation of terrain using PyVista's efficient UniformGrid.

    Args:
        terrain_mask: 3D numpy array where terrain == 0 is solid.
                      Assumed to be in (z, y, x) order.
        ds: Grid spacing (scalar or 3-element array [dx, dy, dz]).
        **kwargs: Additional keyword arguments for plotter.add_mesh.

    Returns:
        pyvista.UnstructuredGrid: A mesh of the terrain voxels.
    """
    if np.all(terrain_mask != 0):
        return None

    # Create dimensions for a grid of (nx+1, ny+1, nz+1) points
    dims = np.array(terrain_mask.shape)[::-1] + 1
    
    if isinstance(ds, (int, float)):
        spacing = (ds, ds, ds)
    else:
        spacing = (ds[0], ds[1], ds[2]) # (dx, dy, dz)
        
    origin = (-spacing[0]/2, -spacing[1]/2, -spacing[2]/2)
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

    # Add the mask as cell data using C-style memory order.
    # This is the key fix to orient the terrain correctly.
    grid.cell_data["terrain"] = terrain_mask.flatten(order='C') # <-- THE FIX IS HERE

    # Extract the cells that represent the terrain (where mask == 0)
    terrain_voxels = grid.threshold(value=0.5, method='lower', scalars="terrain")

    if terrain_voxels.n_cells == 0:
        return None
        
    return terrain_voxels


def visualize_graph_with_terrain_pv(data: Data, channel_idx: int = 0,
                                    slice_axis: str = 'y', slice_value: Optional[float] = None,
                                    show_edges: bool = False, title: str = None):
    """
    Visualize fluid data with terrain and an optional slice using PyVista.
    
    Args:
        data: PyTorch Geometric Data object.
        channel_idx: Index of the channel to visualize.
        slice_axis: Axis for slicing ('x', 'y', or 'z').
        slice_value: Coordinate for the slice. If None, shows a 3D point cloud.
        show_edges: Whether to show graph edges.
        title: Custom plot title.
    """
    # --- Data Preparation ---
    positions = data.pos.numpy()
    values = data.x[:, channel_idx].numpy()
    terrain_mask = data.terrain_mask.numpy()
    ds = data.ds.numpy() if hasattr(data.ds, 'numpy') else data.ds
    channel_name = data.channel_names[channel_idx] if hasattr(data, 'channel_names') else f"Channel {channel_idx}"
    
    # Create a PyVista PolyData object for the fluid cells
    cloud = pv.PolyData(positions)
    cloud[channel_name] = values

    # --- Plotting Setup ---
    plotter = pv.Plotter()

    # Add terrain mesh
    terrain_mesh = create_terrain_voxels_pv(terrain_mask, ds)
    if terrain_mesh is not None:
        plotter.add_mesh(terrain_mesh, color='grey', opacity=0.6)

    # --- Slicing or Full Cloud ---
    if slice_value is not None:
        # Create a slice plane
        slice_plane = cloud.slice(normal=slice_axis, origin=(positions.mean(0)))
        # Adjust the slice position
        slice_plane.origin = slice_plane.center
        if slice_axis == 'x':
            slice_plane.origin[0] = slice_value
        elif slice_axis == 'y':
            slice_plane.origin[1] = slice_value
        else: # 'z'
            slice_plane.origin[2] = slice_value
            
        plotter.add_mesh(slice_plane, scalars=channel_name, cmap='jet', render_points_as_spheres=True, point_size=10)
        plotter.add_mesh(pv.Plane(center=slice_plane.center, direction=slice_axis, 
                                  i_size=np.ptp(positions[:,0]), j_size=np.ptp(positions[:,1])),
                                  color='gray', opacity=0.1)
    else:
        # Show the full 3D point cloud
        plotter.add_mesh(cloud, scalars=channel_name, cmap='jet', render_points_as_spheres=True, point_size=5)

    # --- Edges ---
    if show_edges and hasattr(data, 'edge_index'):
        edge_index = data.edge_index.numpy()
        # Create the line connectivity array for PyVista
        # Format: [2, point1, point2, 2, point3, point4, ...]
        lines = np.c_[np.full(edge_index.shape[1], 2), edge_index.T].flatten()
        edge_mesh = pv.PolyData(positions, lines=lines)
        plotter.add_mesh(edge_mesh, color='gray', opacity=0.3)

    # --- Final Touches ---
    if title is None:
        title = f"Fluid cells: {channel_name}"
        if slice_value is not None:
            title += f" (Slice at {slice_axis}={slice_value:.1f})"
    
    plotter.add_scalar_bar(title=channel_name)
    plotter.show(title=title)


def visualize_vertical_slice_pv(data: Data, channel_idx: int = 0,
                                slice_x: Optional[float] = None,
                                show_terrain: bool = True, title: str = None):
    """
    Show a vertical (y-z) slice through the domain using PyVista.

    Args:
        data: PyTorch Geometric Data object.
        channel_idx: Index of channel to visualize.
        slice_x: X-coordinate for the slice (defaults to middle).
        show_terrain: Whether to show terrain profile.
        title: Custom plot title.
    """
    # --- Data Preparation ---
    positions = data.pos.numpy()
    values = data.x[:, channel_idx].numpy()
    channel_name = data.channel_names[channel_idx] if hasattr(data, 'channel_names') else f"Channel {channel_idx}"
    
    # Default to middle slice if not specified
    if slice_x is None:
        slice_x = (positions[:, 0].min() + positions[:, 0].max()) / 2
    
    # Create a point cloud
    cloud = pv.PolyData(positions)
    cloud[channel_name] = values
    
    # Create the slice
    slice_mesh = cloud.slice(normal='x', origin=[slice_x, 0, 0])

    # --- Plotting Setup ---
    plotter = pv.Plotter()
    
    if slice_mesh.n_points > 0:
        plotter.add_mesh(slice_mesh, scalars=channel_name, cmap='jet',
                         render_points_as_spheres=True, point_size=12)
        plotter.add_scalar_bar(title=channel_name)

    # Show terrain profile
    if show_terrain and hasattr(data, 'terrain_mask'):
        terrain_mask = data.terrain_mask.numpy()
        ds = data.ds.numpy() if hasattr(data.ds, 'numpy') else data.ds
        dx = ds[0] if isinstance(ds, np.ndarray) else ds
        
        terrain_grid = create_terrain_voxels_pv(terrain_mask, ds)
        if terrain_grid is not None:
            terrain_slice = terrain_grid.slice(normal='x', origin=[slice_x, 0, 0])
            plotter.add_mesh(terrain_slice, color='grey')
    
    if title is None:
        title = f"Vertical slice at x={slice_x:.1f}"

    # Set camera view to look down the x-axis for a 2D effect
    plotter.view_yz()
    plotter.enable_parallel_projection()
    plotter.show(title=title)


def quick_pred_vs_true_plot(y, y_pred, pos, terrain_mask, ds, channel_idx=0,
                               slice_axis='y', slice_value=None, channel_name=None,
                               channel_units=None, observed_indices=None):
    # --- Data Preparation ---
    if torch.is_tensor(y): y = y.numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.detach().numpy()
    if torch.is_tensor(pos): pos = pos.numpy()
    if torch.is_tensor(terrain_mask): terrain_mask = terrain_mask.numpy()
    if torch.is_tensor(ds): ds = ds.numpy()
    if observed_indices is not None and torch.is_tensor(observed_indices):
        observed_indices = observed_indices.numpy()

    if channel_name is None: channel_name = f"Channel {channel_idx}"
    ds_scalar = float(ds[0]) if isinstance(ds, np.ndarray) else float(ds)

    units_str = f" [{channel_units}]" if channel_units else ""

    true_values = y[:, channel_idx]
    pred_values = y_pred[:, channel_idx]
    error_values = np.abs(pred_values - true_values)
    mae, rmse = np.mean(error_values), np.sqrt(np.mean(error_values**2))

    cloud = pv.PolyData(pos)
    cloud[f"True - {channel_name}{units_str}"] = true_values
    cloud[f"Predicted - {channel_name}{units_str}"] = pred_values
    cloud[f"Absolute Error{units_str}"] = error_values

    vmin = min(true_values.min(), pred_values.min())
    vmax = max(true_values.max(), pred_values.max())

    # Create plotter with border_visibility off to avoid title overlap
    plotter = pv.Plotter(shape=(1, 3), window_size=(1100, 400), border=False)

    terrain_mesh = create_terrain_voxels_pv(terrain_mask, ds)
    obs_markers = None
    if observed_indices is not None and len(observed_indices) > 0:
        obs_markers = pv.PolyData(pos[observed_indices])

    scalar_names = [f"True - {channel_name}{units_str}", f"Predicted - {channel_name}{units_str}", f"Absolute Error{units_str}"]
    cmaps = ['jet', 'jet', 'hot']
    
    
    for i in range(3):
        plotter.subplot(0, i)
        
        # Add title with proper positioning to avoid overlap with scalar bar

        if terrain_mesh:
            plotter.add_mesh(terrain_mesh.copy(), color='grey', opacity=0.4)
        if obs_markers:
            plotter.add_mesh(obs_markers, color='black', point_size=10, render_points_as_spheres=True)
            plotter.add_mesh(obs_markers, color='white', point_size=7, render_points_as_spheres=True)

        if slice_value is None:
            geom = pv.Cube(x_length=ds[0]*0.95, y_length=ds[1]*0.95, z_length=ds[2]*0.95)
            glyphs = cloud.glyph(scale=False, geom=geom, orient=False)
            plotter.add_mesh(glyphs, scalars=scalar_names[i], cmap=cmaps[i],
                             clim=[vmin, vmax] if i < 2 else None)
        else:
            # Fixed slice visualization
            slice_values_list = slice_value if isinstance(slice_value, (list, np.ndarray)) else [slice_value]
            for val in slice_values_list:
                tolerance = ds_scalar * 0.51
                axis_map = {'x': 0, 'y': 1, 'z': 2}
                axis_idx = axis_map[slice_axis]
                coordinates = cloud.points[:, axis_idx]
                slice_mask = np.abs(coordinates - val) < tolerance
                
                # Only keep points that are in the slice
                if np.any(slice_mask):
                    sliced_points = pos[slice_mask]
                    sliced_data = cloud.point_data[scalar_names[i]][slice_mask]
                    
                    # Create a mesh for just the sliced points
                    slc = pv.PolyData(sliced_points)
                    slc[scalar_names[i]] = sliced_data
                    
                    if slc.n_points > 2:
                        # Create flat squares for slice visualization
                        if slice_axis == 'x':
                            geom = pv.Cube(x_length=ds[0]*0.1, y_length=ds[1], z_length=ds[2])
                        elif slice_axis == 'y':
                            geom = pv.Cube(x_length=ds[0], y_length=ds[1]*0.1, z_length=ds[2])
                        else:  # z
                            geom = pv.Cube(x_length=ds[0], y_length=ds[1], z_length=ds[2]*0.1)
                        
                        glyphs = slc.glyph(scale=False, geom=geom, orient=False)
                        plotter.add_mesh(glyphs, scalars=scalar_names[i], cmap=cmaps[i],
                                         clim=[vmin, vmax] if i < 2 else None)
        

    plotter.link_views()
    plotter.show()
    
    return mae, rmse

def quick_data_plot(y, pos, terrain_mask, ds, channel_idx=0,
                    slice_axis='y', slice_value=None, channel_name=None,
                    channel_units=None, observed_indices=None, 
                    show_stats=True, cmap='jet'):
    """
    Quick visualization of a single channel with terrain.
    
    Args:
        y: Data array of shape (n_points, n_channels)
        pos: Position array of shape (n_points, 3)
        terrain_mask: 3D numpy array where terrain == 0 is solid
        ds: Grid spacing
        channel_idx: Index of channel to visualize
        slice_axis: Axis for slicing ('x', 'y', or 'z')
        slice_value: Coordinate for the slice
        channel_name: Name of the channel
        channel_units: Units for the channel
        observed_indices: Indices of observation points
        show_stats: Whether to show statistics in title
        cmap: Colormap to use
    
    Returns:
        dict: Statistics of the visualized channel
    """
    # --- Data Preparation ---
    if torch.is_tensor(y): y = y.numpy()
    if torch.is_tensor(pos): pos = pos.numpy()
    if torch.is_tensor(terrain_mask): terrain_mask = terrain_mask.numpy()
    if torch.is_tensor(ds): ds = ds.numpy()
    if observed_indices is not None and torch.is_tensor(observed_indices):
        observed_indices = observed_indices.numpy()

    if channel_name is None: 
        channel_name = f"Channel {channel_idx}"
    else:
        units_str = f" [{channel_units}]" if channel_units else ""
        channel_name = f"{channel_name}{units_str}"
    
    ds_scalar = float(ds[0]) if isinstance(ds, np.ndarray) else float(ds)
    
    # Get channel data and compute statistics
    channel_data = y[:, channel_idx]
    stats = {
        'mean': np.mean(channel_data),
        'std': np.std(channel_data),
        'min': np.min(channel_data),
        'max': np.max(channel_data),
        'median': np.median(channel_data)
    }
    
    # Create point cloud
    cloud = pv.PolyData(pos)
    cloud[channel_name] = channel_data

    # Create plotter
    plotter = pv.Plotter(window_size=(800, 700))

    # Create title with statistics
    if show_stats:
        title = f"{channel_name}\n(μ={stats['mean']:.3f}, σ={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}])"
    else:
        title = channel_name
    
    plotter.add_text(title, position=(0.5, 0.95), 
                    font_size=14, viewport=False, font='arial')

    # Add terrain
    terrain_mesh = create_terrain_voxels_pv(terrain_mask, ds)
    if terrain_mesh:
        plotter.add_mesh(terrain_mesh, color='grey', opacity=0.4)
    
    # Add observation markers
    if observed_indices is not None and len(observed_indices) > 0:
        obs_markers = pv.PolyData(pos[observed_indices])
        plotter.add_mesh(obs_markers, color='black', point_size=10, render_points_as_spheres=True)
        plotter.add_mesh(obs_markers, color='white', point_size=7, render_points_as_spheres=True)

    # Add data visualization
    if slice_value is None:
        # 3D cube visualization
        geom = pv.Cube(x_length=ds[1]*0.95, y_length=ds[2]*0.95, z_length=ds[2]*0.95)
        glyphs = cloud.glyph(scale=False, geom=geom, orient=False)
        plotter.add_mesh(glyphs, scalars=channel_name, cmap=cmap)
    else:
        # Slice visualization with proper terrain masking
        tolerance = ds_scalar * 0.1
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        axis_idx = axis_map[slice_axis]
        coordinates = cloud.points[:, axis_idx]
        slice_mask = np.abs(coordinates - slice_value) < tolerance
        
        # Only keep points that are in the slice
        if np.any(slice_mask):
            sliced_points = pos[slice_mask]
            sliced_values = channel_data[slice_mask]
            
            # Create a mesh for just the sliced points
            slc = pv.PolyData(sliced_points)
            slc[channel_name] = sliced_values
            
            if slc.n_points > 2:
                # For 2D visualization, show as flat squares
                if slice_axis == 'x':
                    geom = pv.Cube(x_length=ds[0]*0.1, y_length=ds[1], z_length=ds[2])
                elif slice_axis == 'y':
                    geom = pv.Cube(x_length=ds[0], y_length=ds[1]*0.1, z_length=ds[2])
                else:  # z
                    geom = pv.Cube(x_length=ds[0], y_length=ds[1], z_length=ds[2]*0.1)
                
                glyphs = slc.glyph(scale=False, geom=geom, orient=False)
                plotter.add_mesh(glyphs, scalars=channel_name, cmap=cmap)

    plotter.show()

    return stats


if __name__ == '__main__':
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from datatools import WindTerrainDataset

    parser = argparse.ArgumentParser(description='Plot wind terrain channel from dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to wind terrain dataset (.h5 file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to plot')
    parser.add_argument('--channel', type=int, default=0, help='Channel index to plot (0-7)')
    parser.add_argument('--slice-axis', type=str, default=None, choices=['x', 'y', 'z'],
                       help='Axis to slice along (None for 3D view)')
    parser.add_argument('--slice-value', type=float, default=None,
                       help='Value along slice axis (required if slice-axis is set)')

    args = parser.parse_args()

    # Validate slice parameters
    if args.slice_axis is not None and args.slice_value is None:
        print("Error: --slice-value required when --slice-axis is specified")
        sys.exit(1)

    # Load dataset
    dataset = WindTerrainDataset(
        data_path=args.data,
        mode='eval',
        channels=['ux', 'uy', 'uz', 'p'],
        max_cells_above_terrain=None
    )

    # Get sample
    if args.sample >= len(dataset):
        print(f"Error: Sample index {args.sample} out of range (dataset has {len(dataset)} samples)")
        sys.exit(1)

    data = dataset[args.sample]

    # use quick_data_plot to visualize
    quick_data_plot(
        y=data.x,
        pos=data.pos,
        terrain_mask=data.terrain_mask,
        ds=data.ds,
        channel_idx=args.channel,
        slice_axis=args.slice_axis if args.slice_axis else 'y',
        slice_value=args.slice_value,
        channel_name=dataset.channel_names[args.channel] if hasattr(dataset, 'channel_names') else None,
        channel_units=dataset.channel_units[args.channel] if hasattr(dataset, 'channel_units') else None,
        observed_indices=None,
        show_stats=True,
        cmap='jet'
    )