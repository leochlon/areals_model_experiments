"""
Dataset Statistics

Utilities for computing dataset statistics (mean, std, min, max) for normalization.
Uses iterative computation to handle large datasets efficiently.
"""

import numpy as np
import torch


def compute_dataset_stats(dataloader, device):
    """
    Compute dataset statistics for normalization.
    Uses iterative mean and std computation for memory efficiency.

    Args:
        dataloader: DataLoader for the dataset
        device: Device to use for computations

    Returns:
        dict: Statistics for node features (x), edge attributes, and positions
    """
    
    with torch.no_grad():
        # iterative variables for the x input values
        x_min = []
        x_max = []
        x_sum = torch.zeros(dataloader.dataset.num_node_features, requires_grad=False)
        x_sum_squared = torch.zeros(dataloader.dataset.num_node_features, requires_grad=False)
        n_nodes = 0
        
        # check if the dataset has edge attributes
        num_edge_features = dataloader.dataset.num_edge_features
        if num_edge_features == 0:
            compute_edge_stats = False
        else:
            compute_edge_stats = True
            
            # iterative variables for the edge attributes
            ea_min = []
            ea_max = []
            ea_sum = torch.zeros(num_edge_features, requires_grad=False)
            ea_sum_squared = torch.zeros(num_edge_features, requires_grad=False)
            n_edges = 0

        # check if the dataset has positions
        pos_dim = dataloader.dataset.pos_dim
        if pos_dim == 0:
            compute_pos_stats = False
        else:
            compute_pos_stats = True
            
            # iterative variables for the positions
            pos_min = []
            pos_max = []
            pos_sum = torch.zeros(pos_dim, requires_grad=False)
            pos_sum_squared = torch.zeros(pos_dim, requires_grad=False)
            n_pos = 0

        for data in dataloader:

            # compute batch edge_stats
            if compute_edge_stats:
                ea_min.append(data.edge_attr.min(dim=0).values.tolist())
                ea_max.append(data.edge_attr.max(dim=0).values.tolist())
                ea_sum += data.edge_attr.sum(dim=0)
                ea_sum_squared += (data.edge_attr ** 2).sum(dim=0)
                n_edges += data.edge_attr.shape[0]
            
            # compute batch position stats
            if compute_pos_stats and hasattr(data, 'pos') and data.pos is not None:
                pos_min.append(data.pos.min(dim=0).values.tolist())
                pos_max.append(data.pos.max(dim=0).values.tolist())
                pos_sum += data.pos.sum(dim=0)
                pos_sum_squared += (data.pos ** 2).sum(dim=0)
                n_pos += data.pos.shape[0]

            # compute batch x ignoring the masked input values
            x_min.append(np.nanmin(data.x.numpy(), axis=0).tolist())
            x_max.append(np.nanmax(data.x.numpy(), axis=0).tolist())
            x_sum += np.nansum(data.x.numpy(), axis=0)
            x_sum_squared += np.nansum((data.x ** 2), axis=0)
            n_nodes += data.x.shape[0]

        # save final edge attributes stats
        if compute_edge_stats:
            edge_stats = {'max': torch.tensor(np.max(ea_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                        'min': torch.tensor(np.min(ea_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                        'mean': (ea_sum/n_edges).to(device),
                        'std': torch.sqrt(ea_sum_squared/n_edges - (ea_sum/n_edges)** 2).to(device)}
        else:
            edge_stats = None
        
        # save final position stats
        if compute_pos_stats:
            pos_stats = {'max': torch.tensor(np.max(pos_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                        'min': torch.tensor(np.min(pos_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                        'mean': (pos_sum/n_pos).to(device),
                        'std': torch.sqrt(pos_sum_squared/n_pos - (pos_sum/n_pos)** 2).to(device)}
        else:
            pos_stats = None

        # save final x stats
        x_stats = {'max': torch.tensor(np.max(x_max, axis=0), dtype=torch.float, device=device, requires_grad=False),
                    'min': torch.tensor(np.min(x_min, axis=0), dtype=torch.float, device=device, requires_grad=False),
                    'mean': (x_sum/n_nodes).to(device),
                    'std': torch.sqrt(x_sum_squared/n_nodes - (x_sum/n_nodes)** 2).to(device)}

        return {'x': x_stats, 'edge_attrs': edge_stats, 'pos': pos_stats}
    
def norm_data(data, stats):
    """
    Normalize data using precomputed statistics.

    Args:
        data: Input graph data
        stats: Statistics dict from compute_dataset_stats

    Returns:
        Normalized data
    """
    # normalize x
    data.x = (data.x - stats['x']['mean'].to(data.x.device)) / stats['x']['std'].to(data.x.device)

    # normalize y using x stats
    data.y = (data.y - stats['x']['mean'].to(data.y.device)) / stats['x']['std'].to(data.y.device)

    # normalize edge attributes
    if 'edge_attr' in data and stats['edge_attrs'] is not None:
        data.edge_attr = (data.edge_attr - stats['edge_attrs']['mean'].to(data.edge_attr.device)) / stats['edge_attrs']['std'].to(data.edge_attr.device)
    
    # normalize positions
    if 'pos' in data and stats['pos'] is not None:
        data.pos = (data.pos - stats['pos']['mean'].to(data.pos.device)) / stats['pos']['std'].to(data.pos.device)

    return data

def denorm_data(data, stats):
    """
    Denormalize data using precomputed statistics.

    Args:
        data: Normalized graph data
        stats: Statistics dict from compute_dataset_stats

    Returns:
        Denormalized data
    """
    # denormalize x
    data.x = data.x * stats['x']['std'].to(data.x.device) + stats['x']['mean'].to(data.x.device)

    # denormalize y using x stats
    data.y = data.y * stats['x']['std'].to(data.x.device) + stats['x']['mean'].to(data.x.device)
    
    # denormalize edge attributes
    if 'edge_attr' in data and stats['edge_attrs'] is not None:
        data.edge_attr = data.edge_attr * stats['edge_attrs']['std'].to(data.edge_attr.device) + stats['edge_attrs']['mean'].to(data.edge_attr.device)
    
    # denormalize positions
    if 'pos' in data and stats['pos'] is not None:
        data.pos = data.pos * stats['pos']['std'].to(data.pos.device) + stats['pos']['mean'].to(data.pos.device)

    return data