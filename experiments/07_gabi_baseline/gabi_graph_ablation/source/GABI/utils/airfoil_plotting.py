import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
import numpy as np
from torch_geometric.data import Data
from typing import Optional, Tuple
from scipy.spatial import Delaunay


def set_to_sci_format(ax_axis):
    """Set axis to scientific notation format"""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax_axis.set_major_formatter(formatter)


def get_vmin_vmax(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Get min/max values for consistent color scaling.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values (optional)

    Returns:
        vmin, vmax: Min and max values
    """
    if y_pred is not None:
        vmin = min(np.nanmin(y_true), np.nanmin(y_pred))
        vmax = max(np.nanmax(y_true), np.nanmax(y_pred))
    else:
        vmin = np.nanmin(y_true)
        vmax = np.nanmax(y_true)
    return vmin, vmax


def _constrained_delaunay_triangulation(pos: np.ndarray, node_type: np.ndarray) -> tri.Triangulation:
    """
    Create constrained Delaunay triangulation avoiding all-airfoil triangles.

    Args:
        pos: Node positions (N, 2)
        node_type: Node type indicator (N,) where 1.0 = airfoil, 0.0 = fluid

    Returns:
        matplotlib.tri.Triangulation object with valid triangles
    """
    is_airfoil = (node_type == 1.0)

    # Perform Delaunay triangulation
    delaunay_tri = Delaunay(pos)
    all_triangles = delaunay_tri.simplices

    # Filter out triangles with 3 airfoil nodes (invalid for fluid mesh)
    valid_triangles = []
    for triangle in all_triangles:
        if np.sum(is_airfoil[triangle]) < 3:
            valid_triangles.append(triangle)

    valid_triangles = np.array(valid_triangles)

    # Create matplotlib triangulation with valid triangles
    return tri.Triangulation(pos[:, 0], pos[:, 1], triangles=valid_triangles)


def plot_airfoil_field(data: Data,
                       field: str = 'pressure',
                       plot_predicted: bool = False,
                       plot_diff: bool = False,
                       xlimits: Optional[Tuple[float, float]] = None,
                       ylimits: Optional[Tuple[float, float]] = None,
                       fig_size: Tuple[float, float] = (14, 5),
                       show: bool = True,
                       title: Optional[str] = None) -> plt.Figure:
    """
    Plot airfoil flow field using constrained Delaunay triangulation.

    Args:
        data: PyTorch Geometric Data object with pos, x, y, node_type attributes
        field: Field to plot ('pressure', 'velocity_x', 'velocity_y', 'velocity_mag')
        plot_predicted: Whether to plot predicted field (data.x) alongside ground truth
        plot_diff: Whether to plot difference between prediction and ground truth
        xlimits: Optional (xmin, xmax) limits
        ylimits: Optional (ymin, ymax) limits
        fig_size: Figure size (width, height)
        show: Whether to display the plot
        title: Custom title for the plot

    Returns:
        matplotlib Figure object
    """
    # Move to CPU and convert to numpy
    data = data.cpu()
    pos = data.pos.numpy()
    y_true = data.y.numpy()
    node_type = data.node_type.numpy() if hasattr(data, 'node_type') else None

    # Get field data and metadata
    if field == 'pressure':
        true_values = y_true[:, 0]
        field_label = 'Pressure [Pa]'
        cmap = 'viridis'
        if plot_predicted:
            pred_values = data.x[:, 0].numpy()
    elif field == 'velocity_x':
        true_values = y_true[:, 1]
        field_label = 'X-Velocity [m/s]'
        cmap = 'RdBu_r'
        if plot_predicted:
            pred_values = data.x[:, 1].numpy()
    elif field == 'velocity_y':
        true_values = y_true[:, 2]
        field_label = 'Y-Velocity [m/s]'
        cmap = 'RdBu_r'
        if plot_predicted:
            pred_values = data.x[:, 2].numpy()
    elif field == 'velocity_mag':
        true_values = np.linalg.norm(y_true[:, 1:3], axis=1)
        field_label = 'Velocity Magnitude [m/s]'
        cmap = 'plasma'
        if plot_predicted:
            pred_values = np.linalg.norm(data.x[:, 1:3].numpy(), axis=1)
    else:
        raise ValueError(f"Unknown field: {field}")

    # Create constrained Delaunay triangulation (or standard if no node_type)
    if node_type is not None:
        triangulation = _constrained_delaunay_triangulation(pos, node_type)
    else:
        triangulation = tri.Triangulation(pos[:, 0], pos[:, 1])

    # Set up figure
    ncols = 1 + (1 if plot_predicted else 0) + (1 if plot_diff else 0)
    fig, axes = plt.subplots(1, ncols, figsize=fig_size)
    if ncols == 1:
        axes = [axes]

    # Get color scale limits
    if plot_predicted:
        vmin, vmax = get_vmin_vmax(true_values, pred_values)
    else:
        vmin, vmax = get_vmin_vmax(true_values)

    # Plot ground truth
    ax_idx = 0
    tpc = axes[ax_idx].tripcolor(triangulation, true_values, shading='flat',
                                   vmin=vmin, vmax=vmax, cmap=cmap)
    axes[ax_idx].triplot(triangulation, 'k-', alpha=0.1, linewidth=0.3)
    axes[ax_idx].set_title('Ground Truth')
    axes[ax_idx].set_aspect('equal')
    cb = plt.colorbar(tpc, ax=axes[ax_idx])
    cb.set_label(field_label, fontsize=12)
    set_to_sci_format(cb.ax.yaxis)

    # Plot prediction if requested
    if plot_predicted:
        ax_idx += 1
        tpc_p = axes[ax_idx].tripcolor(triangulation, pred_values, shading='flat',
                                         vmin=vmin, vmax=vmax, cmap=cmap)
        axes[ax_idx].triplot(triangulation, 'k-', alpha=0.1, linewidth=0.3)
        axes[ax_idx].set_title('Predicted')
        axes[ax_idx].set_aspect('equal')
        cb_p = plt.colorbar(tpc_p, ax=axes[ax_idx])
        cb_p.set_label(field_label, fontsize=12)
        set_to_sci_format(cb_p.ax.yaxis)

        # Plot difference if requested
        if plot_diff:
            ax_idx += 1
            diff_values = true_values - pred_values
            tpc_d = axes[ax_idx].tripcolor(triangulation, diff_values, shading='flat',
                                             cmap='seismic')
            axes[ax_idx].triplot(triangulation, 'k-', alpha=0.1, linewidth=0.3)
            axes[ax_idx].set_title('Difference (Truth - Pred)')
            axes[ax_idx].set_aspect('equal')
            cb_d = plt.colorbar(tpc_d, ax=axes[ax_idx])
            cb_d.set_label(f'{field_label} Diff', fontsize=12)
            set_to_sci_format(cb_d.ax.yaxis)

    # Set axis limits for all subplots
    for ax in axes:
        if xlimits is not None:
            ax.set_xlim(xlimits)
        if ylimits is not None:
            ax.set_ylim(ylimits)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    # Add overall title
    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_surface_pressure(data: Data,
                          plot_predicted: bool = False,
                          fig_size: Tuple[float, float] = (12, 4),
                          show: bool = True,
                          title: Optional[str] = None) -> plt.Figure:
    """
    Plot pressure distribution on airfoil surface.

    Args:
        data: PyTorch Geometric Data object
        plot_predicted: Whether to plot predicted alongside ground truth
        fig_size: Figure size (width, height)
        show: Whether to display the plot
        title: Custom title

    Returns:
        matplotlib Figure object
    """
    # Move to CPU
    data = data.cpu()

    # Get airfoil nodes
    airfoil_mask = (data.node_type == 1).numpy()
    pos = data.pos[airfoil_mask].numpy()
    p_true = data.y[airfoil_mask, 0].numpy()

    # Sort by x-coordinate for better visualization
    sort_idx = np.argsort(pos[:, 0])
    x_sorted = pos[sort_idx, 0]
    p_true_sorted = p_true[sort_idx]

    # Set up figure
    ncols = 2 if plot_predicted else 1
    fig, axes = plt.subplots(1, ncols, figsize=fig_size)
    if ncols == 1:
        axes = [axes]

    # Plot ground truth
    axes[0].scatter(pos[:, 0], pos[:, 1], c=p_true, cmap='viridis', s=20)
    axes[0].plot(x_sorted, p_true_sorted, 'k-', alpha=0.3, linewidth=1)
    axes[0].set_title('Ground Truth Surface Pressure')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[0].set_aspect('equal')

    # Plot prediction if requested
    if plot_predicted:
        p_pred = data.x[airfoil_mask, 0].numpy()
        p_pred_sorted = p_pred[sort_idx]

        axes[1].scatter(pos[:, 0], pos[:, 1], c=p_pred, cmap='viridis', s=20)
        axes[1].plot(x_sorted, p_pred_sorted, 'k-', alpha=0.3, linewidth=1)
        axes[1].set_title('Predicted Surface Pressure')
        axes[1].set_xlabel('x [m]')
        axes[1].set_ylabel('y [m]')
        axes[1].set_aspect('equal')

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def quick_pred_vs_true_plot(data: Data,
                            field: str = 'pressure',
                            xlimits: Optional[Tuple[float, float]] = None,
                            ylimits: Optional[Tuple[float, float]] = None,
                            show: bool = True) -> Tuple[float, float]:
    """
    Quick comparison plot of prediction vs ground truth with error metrics.

    Args:
        data: PyTorch Geometric Data object with x (prediction) and y (ground truth)
        field: Field to plot
        xlimits: Optional x-axis limits
        ylimits: Optional y-axis limits
        show: Whether to display the plot

    Returns:
        mae, rmse: Mean absolute error and root mean squared error
    """
    # Plot with all three panels
    fig = plot_airfoil_field(data, field=field, plot_predicted=True, plot_diff=True,
                             xlimits=xlimits, ylimits=ylimits, show=False)

    # Compute error metrics
    y_true = data.y.numpy()
    y_pred = data.x.numpy()

    if field == 'pressure':
        true_vals = y_true[:, 0]
        pred_vals = y_pred[:, 0]
    elif field == 'velocity_x':
        true_vals = y_true[:, 1]
        pred_vals = y_pred[:, 1]
    elif field == 'velocity_y':
        true_vals = y_true[:, 2]
        pred_vals = y_pred[:, 2]
    elif field == 'velocity_mag':
        true_vals = np.linalg.norm(y_true[:, 1:3], axis=1)
        pred_vals = np.linalg.norm(y_pred[:, 1:3], axis=1)

    error = np.abs(true_vals - pred_vals)
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error**2))

    # Add error metrics to title
    fig.suptitle(f'{field.title()} - MAE: {mae:.3e}, RMSE: {rmse:.3e}',
                 fontsize=14, y=0.98)

    if show:
        plt.show()

    return mae, rmse


if __name__ == '__main__':
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from datatools import AirfoilDataset

    parser = argparse.ArgumentParser(description='Plot airfoil flow field from dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to airfoil dataset (.zip file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to plot')
    parser.add_argument('--field', type=str, default='pressure',
                       choices=['pressure', 'velocity_x', 'velocity_y', 'velocity_mag'],
                       help='Field to plot')
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help='X-axis limits (xmin xmax)')
    parser.add_argument('--ylim', type=float, nargs=2, default=None, help='Y-axis limits (ymin ymax)')

    args = parser.parse_args()

    # Load dataset
    dataset = AirfoilDataset(data_path=args.data)

    # Get sample
    if args.sample >= len(dataset):
        print(f"Error: Sample index {args.sample} out of range (dataset has {len(dataset)} samples)")
        sys.exit(1)

    data = dataset[args.sample]

    # Plot
    xlimits = tuple(args.xlim) if args.xlim else None
    ylimits = tuple(args.ylim) if args.ylim else None

    plot_airfoil_field(data, field=args.field, xlimits=xlimits, ylimits=ylimits,
                      title=f'Sample {args.sample} - {args.field}', show=True)
