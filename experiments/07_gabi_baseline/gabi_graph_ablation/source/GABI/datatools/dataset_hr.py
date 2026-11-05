"""
Heat Rectangle Dataset

PyTorch Geometric dataset for 2D heat equation simulations on rectangular meshes.
Data is stored in pickle format containing pre-generated triangular meshes.
"""

import torch
from torch_geometric.data import Dataset, Data
import os
import pickle


class HeatRectangleDataset(Dataset):
    """Dataset for steady-state heat equation on rectangular meshes"""

    def __init__(self, data_path: str, transform=None, pre_transform=None):
        """
        Args:
            data_path: Path to pickle file containing simulation graphs
        """
        super().__init__(None, transform, pre_transform)
        self.__data_path = data_path

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"pkl file not found at {data_path}")

        # load in the pkl file (small file, so we can load it all into memory)
        try:
            with open(data_path, 'rb') as f:
                self.data_list = pickle.load(f)
        except Exception as e:
            raise IOError(f"Failed to read pkl file '{data_path}': {e}")
        self.__num_graphs = len(self.data_list)

    def _compute_boundary_nodes(self, data: Data) -> torch.Tensor:
        """
        Compute boundary nodes from the mesh connectivity.
        Returns a boolean tensor where True indicates a boundary node.
        
        For triangular meshes, boundary nodes are those that belong to edges 
        that are only part of one triangle.
        """
        faces = data.face  # Shape: [3, num_faces]
        num_nodes = data.num_nodes
        
        # Vectorized approach: create all edges from faces at once
        # Each face contributes 3 edges
        edges = torch.cat([
            torch.stack([faces[0], faces[1]], dim=0),  # edge 0-1
            torch.stack([faces[1], faces[2]], dim=0),  # edge 1-2
            torch.stack([faces[2], faces[0]], dim=0),  # edge 2-0
        ], dim=1)  # Shape: [2, 3*num_faces]
        
        # Sort edges to handle undirected nature (smaller index first)
        edges_sorted = torch.sort(edges, dim=0)[0]
        
        # Convert to unique edges and count occurrences
        # Encode edges as single numbers for efficient counting
        # This assumes node indices fit in int32
        edge_codes = edges_sorted[0] * num_nodes + edges_sorted[1]
        
        # Count occurrences of each edge
        unique_codes, counts = torch.unique(edge_codes, return_counts=True)
        
        # Boundary edges appear only once
        boundary_edge_codes = unique_codes[counts == 1]
        
        # Decode back to node pairs
        boundary_edges = torch.stack([
            boundary_edge_codes // num_nodes,  # first node
            boundary_edge_codes % num_nodes    # second node
        ], dim=0)
        
        # Mark boundary nodes
        is_boundary = torch.zeros(num_nodes, dtype=torch.bool)
        is_boundary[boundary_edges.flatten()] = True
        
        return is_boundary

    def _get_boundary_encoding(self, data: Data) -> Data:
        """
        Add boundary node one-hot encoding to the node features.
        """
        is_boundary = self._compute_boundary_nodes(data)
        
        # Create one-hot encoding: [is_interior, is_boundary]
        boundary_encoding = torch.zeros((data.num_nodes, 2))
        boundary_encoding[~is_boundary, 0] = 1  # Interior nodes
        boundary_encoding[is_boundary, 1] = 1   # Boundary nodes
        
        return boundary_encoding

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self.__num_graphs:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self.__num_graphs}")
        data = self.data_list[idx].clone()  # clone to avoid modifying original data
        data.edge_attr = None  # remove edge attributes if any from previous processing
        
        # create the target node features, which are the same as the input node features
        if data.y is None:
            data.y = data.x.clone()
        
        # get the boundary encoding
        data.boundary_encoding = self._get_boundary_encoding(data)
        
        
        if self.transform:
            data = self.transform(data)
        
        return data
        
    @property
    def num_node_features(self) -> int:
        if self.__num_graphs == 0: return 0
        data = self.__getitem__(0)
        if isinstance(data, tuple):
            data = data[0]
        return data.x.shape[1]

    @property
    def num_node_output_features(self) -> int:
        if self.__num_graphs == 0: return 0
        data = self.__getitem__(0)
        if isinstance(data, tuple):
            data = data[0]
        return data.y.shape[1]

    @property
    def num_edge_features(self) -> int:
        if self.__num_graphs == 0: return 0
        data = self.__getitem__(0)
        if isinstance(data, tuple):
            data = data[0]
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            return 0
        if len(data.edge_attr.shape) < 2:
            return 0
        return data.edge_attr.shape[1]

    @property
    def pos_dim(self) -> int:
        """Number of position coordinates"""
        if len(self) == 0:
            return 0
        data = self.__getitem__(0)
        if isinstance(data, tuple):
            data = data[0]
        if hasattr(data, 'pos') and data.pos is not None:
            return data.pos.shape[1]
        return 0
    
    def __len__(self):
        return self.__num_graphs
    
    
    def get_data_dims_dict(self) -> dict:
        """ Returns a dictionary with the number of features for each type of data."""
        data = self.__getitem__(0)
        data = data[0] if isinstance(data, tuple) else data

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


