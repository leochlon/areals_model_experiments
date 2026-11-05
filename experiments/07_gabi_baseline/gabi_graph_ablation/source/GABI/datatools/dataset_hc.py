"""
Helmholtz Car Dataset

PyTorch Geometric dataset for Helmholtz equation solutions on car meshes.
Data is stored in pickle format containing pre-solved triangular surface meshes.
"""

import torch
from torch_geometric.data import Dataset, Data
import os
import pickle


class HelmholtzCarDataset(Dataset):
    """Dataset for Helmholtz equation solutions on 3D car meshes"""

    def __init__(self, data_path: str, transform=None, pre_transform=None):
        """
        Args:
            data_path: Path to pickle file containing Helmholtz solutions
        """
        super().__init__(None, transform, pre_transform)
        self.__data_path = data_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Pickle file not found at {data_path}")

        # Load data
        with open(data_path, 'rb') as f:
            self.__data_list = pickle.load(f)

        if not isinstance(self.__data_list, list):
            raise ValueError("Pickle file must contain a list of Data objects")

        print(f"Loaded {len(self.__data_list)} Helmholtz car samples from {data_path}")

    def len(self):
        return len(self.__data_list)

    def get(self, idx: int) -> Data:
        if not 0 <= idx < len(self.__data_list):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.__data_list)} items")

        data = self.__data_list[idx].clone()

        # Ensure data has required attributes
        if not hasattr(data, 'pos'):
            raise ValueError(f"Sample {idx} missing 'pos' attribute")
        if not hasattr(data, 'face'):
            raise ValueError(f"Sample {idx} missing 'face' attribute")
        if not hasattr(data, 'x'):
            raise ValueError(f"Sample {idx} missing 'x' (solution) attribute")

        # Set target (for autoencoder, y = x)
        if not hasattr(data, 'y') or data.y is None:
            data.y = data.x.clone()

        # Create edge_index from faces if not present
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            # Extract edges from triangular faces
            # Each triangle [v0, v1, v2] contributes 3 edges: (v0,v1), (v1,v2), (v2,v0)
            faces = data.face  # Shape: [3, num_faces]

            edges = torch.cat([
                torch.stack([faces[0], faces[1]], dim=0),  # edge v0-v1
                torch.stack([faces[1], faces[2]], dim=0),  # edge v1-v2
                torch.stack([faces[2], faces[0]], dim=0),  # edge v2-v0
            ], dim=1)  # Shape: [2, 3*num_faces]

            # Remove duplicate edges (undirected graph)
            # Sort each edge so smaller index comes first
            edges_sorted = torch.sort(edges, dim=0)[0]
            # Get unique edges
            edges_unique = torch.unique(edges_sorted, dim=1)
            # Create bidirectional edges
            data.edge_index = torch.cat([edges_unique, edges_unique.flip(0)], dim=1)

        # Create empty edge attributes
        data.edge_attr = None 

        # Helmholtz Car dataset does not use boundary encoding
        # Set to empty tensor with shape (num_nodes, 0)
        num_nodes = data.pos.shape[0]
        data.boundary_encoding = torch.zeros((num_nodes, 0), dtype=torch.float)

        return data

    @property
    def num_node_features(self) -> int:
        """Number of input node features"""
        if len(self) == 0:
            return 0
        data = self.get(0)
        return data.x.shape[1]

    @property
    def num_node_output_features(self) -> int:
        """Number of output node features"""
        if len(self) == 0:
            return 0
        data = self.get(0)
        return data.y.shape[1]

    @property
    def num_edge_features(self) -> int:
        """Number of edge features"""
        if len(self) == 0:
            return 0
        data = self.get(0)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and len(data.edge_attr.shape) > 1:
            return data.edge_attr.shape[1]
        return 0

    @property
    def pos_dim(self) -> int:
        """Number of position coordinates"""
        if len(self) == 0:
            return 0
        data = self.get(0)
        if hasattr(data, 'pos') and data.pos is not None:
            return data.pos.shape[1]
        return 0

    def get_data_dims_dict(self) -> dict:
        """Returns a dictionary with the number of features for each type of data"""
        if len(self) == 0:
            return {'node_feature_dim': 0, 'edge_feature_dim': 0, 'node_out_dim': 0}

        # Get raw data without transform to avoid issues with dimension checking
        data = self.get(0)

        edge_feature_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and len(data.edge_attr.shape) > 1:
            edge_feature_dim = data.edge_attr.shape[1]

        return {
            'node_feature_dim': data.x.shape[1],
            'edge_feature_dim': edge_feature_dim,
            'node_out_dim': data.y.shape[1],
            'pos_dim': data.pos.shape[1],
            'be_dim': data.boundary_encoding.shape[1]
        }
