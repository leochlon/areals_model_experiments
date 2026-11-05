"""
Transformer-based Geometric Autoencoder
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, Linear
from torch_geometric.utils import to_dense_batch


class TransformerGeomAutoencoder(torch.nn.Module):
    """Geometric autoencoder using Transformer layers"""

    def __init__(self, **kwargs):
        super().__init__()
        required_keys = {'node_feature_dim', 'latent_dim', 'z_dim', 'use_boundary_encoding',
                        'use_pos', 'be_dim', 'pos_dim', 'n_layers', 'n_heads'}
        assert required_keys.issubset(kwargs), f"Missing keys: {required_keys - set(kwargs.keys())}"

        additional_inputs = kwargs['be_dim'] if kwargs['use_boundary_encoding'] else 0
        additional_inputs += kwargs['pos_dim'] if kwargs['use_pos'] else 0

        # Input lifting MLP
        self.node_lifter = MLP(
            in_channels=kwargs['node_feature_dim'] + additional_inputs,
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['latent_dim'],
            num_layers=2,
            act='relu',
            norm='layer'
        )

        # Create transformer encoder with layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=kwargs['latent_dim'],
            nhead=kwargs['n_heads'],
            dim_feedforward=kwargs['latent_dim'],
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoding_layers = nn.TransformerEncoder(encoder_layer, num_layers=kwargs['n_layers'])

        # Project to latent z
        self.to_z = Linear(kwargs['latent_dim'], kwargs['z_dim'])

        # Project from latent z
        self.from_z = Linear(kwargs['z_dim'] + additional_inputs, kwargs['latent_dim'])

        # Create transformer decoder with layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=kwargs['latent_dim'],
            nhead=kwargs['n_heads'],
            dim_feedforward=kwargs['latent_dim'],
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.decoding_layers = nn.TransformerEncoder(decoder_layer, num_layers=kwargs['n_layers'])

        # Output projection MLP
        self.output_projector = MLP(
            in_channels=kwargs['latent_dim'],
            hidden_channels=kwargs['latent_dim'],
            out_channels=kwargs['node_feature_dim'],
            num_layers=2,
            act='relu',
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
            data: Graph data with node features and batch information

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

        # Convert to dense batch with mask
        x_dense, mask = to_dense_batch(x, data.batch)
        # x_dense: [batch_size, max_num_nodes, latent_dim]
        # mask: [batch_size, max_num_nodes]

        # Create attention mask for transformer (True = ignore padding)
        src_key_padding_mask = ~mask

        # Encode through transformer layers with masking
        x_encoded = self.encoding_layers(x_dense, src_key_padding_mask=src_key_padding_mask)

        # Pool over valid nodes (mean pooling)
        x_masked = x_encoded * mask.unsqueeze(-1)  # Zero out padding
        lengths = mask.sum(dim=1, keepdim=True)  # Number of valid nodes per graph
        x_pooled = x_masked.sum(dim=1) / lengths  # Mean over valid nodes

        # Project to z
        z = self.to_z(x_pooled)

        return z

    def decode(self, z_samples, data_batch):
        """
        Decode latent samples to node features.

        Args:
            z_samples: Latent samples [batch_size, z_dim]
            data_batch: Batch of graph data

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

        # Convert to dense batch for decoding
        x_dense, mask = to_dense_batch(x, data_batch.batch)

        # Create attention mask for decoder
        src_key_padding_mask = ~mask

        # Pass through decoder layers with masking
        x_decoded = self.decoding_layers(x_dense, src_key_padding_mask=src_key_padding_mask)

        # Convert back to flat tensor (only keep valid nodes)
        batch_size = x_dense.size(0)
        x_list = []
        for i in range(batch_size):
            valid_nodes = mask[i].sum().item()
            x_list.append(x_decoded[i, :valid_nodes])

        x = torch.cat(x_list, dim=0)

        # Project back to output space
        x = self.output_projector(x)

        return x
