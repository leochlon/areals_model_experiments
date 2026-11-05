"""
GCN-based Geometric Autoencoder
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP, Linear
from torch_geometric.nn import pool
from torch.nn import ModuleList


class GCNGeomAutoencoder(torch.nn.Module):
    """Geometric autoencoder using GCN layers"""

    def __init__(self, **kwargs):
        super().__init__()
        required_keys = {'node_feature_dim', 'edge_feature_dim', 'latent_dim', 'z_dim', 'n_layers',
                        'use_boundary_encoding', 'be_dim', 'use_pos', 'pos_dim'}
        assert required_keys.issubset(kwargs), f"Missing keys: {required_keys - set(kwargs.keys())}"

        n_layers = kwargs['n_layers']

        additional_inputs = kwargs['be_dim'] if kwargs['use_boundary_encoding'] else 0
        additional_inputs += kwargs['pos_dim'] if kwargs['use_pos'] else 0

        # Input lifting MLP
        self.node_lifter = MLP(
            in_channels=kwargs['node_feature_dim'] + additional_inputs,
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['latent_dim'],
            num_layers=2,
            act='silu',
            norm='layer'
        )

        # Encoding layers
        self.encoding_layers = ModuleList([
            GCNConv(kwargs['latent_dim'] * 2, kwargs['latent_dim'])
            for _ in range(n_layers)
        ])

        # Project to latent z
        self.to_z = Linear(kwargs['latent_dim'], kwargs['z_dim'])

        # Project from latent z
        self.from_z = Linear(kwargs['z_dim'] + additional_inputs, kwargs['latent_dim'])

        # Decoding layers
        self.decoding_layers = ModuleList([
            GCNConv(kwargs['latent_dim'] * 2, kwargs['latent_dim'])
            for _ in range(n_layers)
        ])

        # Output projection MLP
        self.output_projector = MLP(
            in_channels=kwargs['latent_dim'],
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['node_feature_dim'],
            num_layers=2,
            act='silu',
            norm=None,
            plain_last=True
        )

        self.z_dim = kwargs['z_dim']
        self.use_boundary_encoding = kwargs['use_boundary_encoding']
        self.use_pos = kwargs['use_pos']

    def forward(self, data):
        # Encode input to latent representation
        data.z = self.encode(data)

        # Decode latent back to physical space
        data.x = self.decode(data.z, data)

        return data

    def encode(self, data):
        """
        Encode input data to latent representation.

        Args:
            data: Graph data with node features, edges, and batch information

        Returns:
            Latent representation z [batch_size, z_dim]
        """
        # Concatenate additional inputs
        x = data.x
        if self.use_boundary_encoding:
            x = torch.cat([x, data.boundary_encoding], dim=1)
        if self.use_pos:
            x = torch.cat([x, data.pos], dim=1)

        # Lift input to latent space
        x = self.node_lifter(x)

        # Encode through GCN layers with global pooling
        # Note: GCN doesn't use edge features - edge_weight is kept as None
        # Using normalized edge features as weights causes numerical instability
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x, data.batch)
            x_expanded = x_global[data.batch]
            x = F.silu(conv(torch.cat([x, x_expanded], dim=1), data.edge_index, edge_weight=None))

        # Pool and project to z
        x_pooled = pool.global_mean_pool(x, data.batch)
        z = self.to_z(x_pooled)

        return z

    def decode(self, z_samples, data_batch):
        """
        Decode latent samples to node features.

        Args:
            z_samples: Latent samples [batch_size, z_dim]
            data_batch: Batch of graph data with edge information

        Returns:
            Decoded node features [num_nodes_total, node_feature_dim]
        """
        # Expand z to all nodes
        x = z_samples[data_batch.batch]

        # Concatenate with additional inputs
        if self.use_boundary_encoding:
            x = torch.cat([x, data_batch.boundary_encoding], dim=1)
        if self.use_pos:
            x = torch.cat([x, data_batch.pos], dim=1)

        # Project from z to latent dimension
        x = self.from_z(x)

        # Decode through GCN layers with global pooling
        # Note: GCN doesn't use edge features - edge_weight is kept as None
        # Using normalized edge features as weights causes numerical instability
        for conv in self.decoding_layers:
            x_global = pool.global_mean_pool(x, data_batch.batch)
            x_expanded = x_global[data_batch.batch]
            x = F.silu(conv(torch.cat([x, x_expanded], dim=1), data_batch.edge_index, edge_weight=None))

        # Project back to output space
        x = self.output_projector(x)

        return x
