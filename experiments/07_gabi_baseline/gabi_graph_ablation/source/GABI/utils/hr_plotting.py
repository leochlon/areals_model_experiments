
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_hr_graph(data, u, ax, ObsIdx=None, title=None):
    points = data.pos.cpu().detach().numpy()
    faces = data.face.cpu().detach().numpy()

    # Plot filled contours
    contourf = ax.tricontourf(points[:, 0], points[:, 1], faces.T, u[:], levels=20, cmap='viridis')
    plt.colorbar(contourf, ax=ax)

    # Overlay contour lines (iso-contours)
    contour = ax.tricontour(points[:, 0], points[:, 1], faces.T, u[:], levels=20, colors='k', linewidths=0.5)

    # Draw mesh
    faces_torch = torch.tensor(faces.T, dtype=torch.long).T
    tri_plot = mtri.Triangulation(points[:, 0], points[:, 1], triangles=faces.T)
    ax.triplot(tri_plot, color='white', alpha=0.2)

    # Plot observed points if provided
    if ObsIdx is not None:
        plt.scatter(
            data.pos[:, 0][ObsIdx].detach().cpu().numpy(),
            data.pos[:, 1][ObsIdx].detach().cpu().numpy(),
            c='r', s=10, alpha=1.0
        )

    # Final formatting
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    if title:
        ax.set_title(title)


if __name__ == '__main__':
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from datatools import HeatRectangleDataset

    parser = argparse.ArgumentParser(description='Plot heat rectangle temperature field from dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to heat rectangle dataset (.pkl file)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to plot')
    parser.add_argument('--field', type=str, default='temperature',
                       choices=['temperature'],
                       help='Field to plot (currently only temperature)')

    args = parser.parse_args()

    # Load dataset
    dataset = HeatRectangleDataset(data_path=args.data)

    # Get sample
    if args.sample >= len(dataset):
        print(f"Error: Sample index {args.sample} out of range (dataset has {len(dataset)} samples)")
        sys.exit(1)

    data = dataset[args.sample]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Extract temperature field (assuming it's in data.y)
    u = data.y[:, 0].cpu().detach().numpy()

    plot_hr_graph(data, u, ax, title=f'Sample {args.sample} - Temperature')
    plt.tight_layout()
    plt.show()
