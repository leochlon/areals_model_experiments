"""
GEN-based Geometric Autoencoder
"""

import torch
from torch_geometric.nn import pool
from torch.nn import ModuleList
from torch_geometric.nn import GENConv, MLP, Linear


class GENGeomAutoencoder(torch.nn.Module):
    """Geometric autoencoder using GEN layers"""

    def __init__(self, **kwargs):
        super().__init__()
        required_keys = {'node_feature_dim', 'edge_feature_dim', 'latent_dim', 'z_dim', 'n_layers',
                        'use_boundary_encoding', 'be_dim', 'use_pos', 'pos_dim'}
        assert required_keys.issubset(kwargs), f"Missing keys: {required_keys - set(kwargs.keys())}"

        n_layers = kwargs['n_layers']

        self.training_noise = kwargs.get('training_noise', 0.0)

        additional_inputs = kwargs['be_dim'] if kwargs['use_boundary_encoding'] else 0
        additional_inputs += kwargs['pos_dim'] if kwargs['use_pos'] else 0

        # Input lifting MLPs for nodes and edges
        self.node_lifter = MLP(
            in_channels=kwargs['node_feature_dim'] + additional_inputs,
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['latent_dim'],
            num_layers=2,
            act='relu',
            norm='layer'
        )

        # Edge features are optional
        self.has_edge_features = kwargs['edge_feature_dim'] > 0
        if self.has_edge_features:
            self.edge_lifter = MLP(
                in_channels=kwargs['edge_feature_dim'],
                hidden_channels=kwargs['latent_dim'],
                out_channels=kwargs['latent_dim'],
                num_layers=2,
                act='relu',
                norm='layer'
            )
        else:
            self.edge_lifter = None

        # Encoding layers
        self.encoding_layers = ModuleList([
            GENConv(kwargs['latent_dim'] * 2, kwargs['latent_dim'], norm='layer')
            for _ in range(n_layers)
        ])

        # Project to latent z
        self.to_z = Linear(kwargs['latent_dim'], kwargs['z_dim'])

        # Edge projection for decoder (only if edge features exist)
        if self.has_edge_features:
            self.edge_projector = MLP(
                in_channels=kwargs['edge_feature_dim'],
                hidden_channels=kwargs['latent_dim'],
                out_channels=kwargs['latent_dim'],
                num_layers=2,
                act='relu',
                norm='layer'
            )
        else:
            self.edge_projector = None

        # Project from latent z
        self.from_z = Linear(kwargs['z_dim'] + additional_inputs, kwargs['latent_dim'])

        # Decoding layers
        self.decoding_layers = ModuleList([
            GENConv(kwargs['latent_dim'] * 2, kwargs['latent_dim'], norm='layer')
            for _ in range(n_layers)
        ])

        # Output projection MLP
        self.output_projector = MLP(
            in_channels=kwargs['latent_dim'],
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['node_feature_dim'],
            num_layers=2,
            act='relu',
            norm='layer',
            plain_last=True,
            dropout=0.0
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

        # Add noise during training for regularization
        if self.training and self.training_noise > 0:
            x += torch.randn_like(x) * self.training_noise

        # Lift input to latent space
        x = self.node_lifter(x)
        edge_attr = self.edge_lifter(data.edge_attr) if self.has_edge_features else None

        # Encode through GEN layers with global pooling
        for conv in self.encoding_layers:
            x_global = pool.global_mean_pool(x, data.batch)
            x_expanded = x_global[data.batch]
            x += conv(torch.cat([x, x_expanded], dim=1), data.edge_index, edge_attr=edge_attr)

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

        # Add noise during training for regularization
        if self.training and self.training_noise > 0:
            x += torch.randn_like(x) * self.training_noise

        # Project from z to latent dimension
        x = self.from_z(x)

        # Project edge attributes to latent space
        edge_attr = self.edge_projector(data_batch.edge_attr) if self.has_edge_features else None

        # Decode through GEN layers with global pooling
        for conv in self.decoding_layers:
            x_global = pool.global_mean_pool(x, data_batch.batch)
            x_expanded = x_global[data_batch.batch]
            x += conv(torch.cat([x, x_expanded], dim=1), data_batch.edge_index, edge_attr=edge_attr)

        # Project back to output space
        x = self.output_projector(x)

        return x
