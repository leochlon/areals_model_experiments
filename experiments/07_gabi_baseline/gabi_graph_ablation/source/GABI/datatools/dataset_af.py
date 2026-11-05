"""
Airfoil Dataset

PyTorch Geometric dataset for 2D CFD simulations around airfoils.
Data from OpenFOAM simulations stored in compressed zip format.
"""

from torch_geometric.data import Dataset
import torch
from zipfile import ZipFile
import io


class AirfoilDataset(Dataset):
    """Dataset for 2D OpenFOAM CFD simulations around airfoils"""

    def __init__(self, data_path: str,
                 use_boundary_encoding: bool = True,
                 transform=None, pre_transform=None):
        """
        Args:
            data_path: Path to zip file containing CFD simulation graphs
            use_boundary_encoding: Add boundary encoding based on node type (airfoil vs fluid)
        """
        super().__init__(None, transform, pre_transform)
        self.__data_path = data_path
        with ZipFile(data_path, 'r') as zf:
            self.__num_graphs = len(zf.namelist())

        self.use_boundary_encoding = use_boundary_encoding

    def __getitem__(self, idx):
        # read the zip file and select the data to load in by index
        with ZipFile(self.__data_path, 'r') as zf:
            with zf.open(zf.namelist()[idx]) as item:
                stream = io.BytesIO(item.read())
                data = torch.load(stream, weights_only=False)
        
        # make sure features are float
        data.x = data.x.float()
        data.pos = data.pos.float()

        # Keep pressure and velocity features (drop sdf if present)
        # Note: data.x contains [pressure, ux, uy, sdf] at this point
        if data.x.shape[1] >= 4:
            data.x = data.x[:, [0, 1, 2]]  # Keep pressure and velocities

        # Remove extra attributes from original data that aren't needed for training
        # These have edge-level or face-level dimensions that cause batching issues
        attrs_to_remove = ['triangles', 'triangle_points', 'globals',
                          'global_feat_labels', 'node_feat_labels', 'edge_feat_labels']
        for attr in attrs_to_remove:
            if hasattr(data, attr):
                delattr(data, attr)

        # Add boundary encoding if enabled
        if self.use_boundary_encoding:
            # Create boundary encoding: [is_fluid, is_airfoil]
            num_nodes = data.num_nodes
            boundary_encoding = torch.zeros((num_nodes, 2), dtype=torch.float)
            boundary_encoding[data.node_type == 0, 0] = 1.0  # Fluid nodes
            boundary_encoding[data.node_type == 1, 1] = 1.0  # Airfoil nodes
            data.boundary_encoding = boundary_encoding
        else:
            # No boundary encoding - empty tensor
            data.boundary_encoding = torch.zeros((data.num_nodes, 0), dtype=torch.float)

        # Set target to match current input features (after all modifications)
        data.y = data.x.clone()
        
        # remove previous edge attributes if any (transform handles this)
        data.edge_attr = None

        # Apply transform if specified (e.g., Cartesian for edge features)
        if self.transform is not None:
            data = self.transform(data)

        return data

    @property
    def num_node_output_features(self) -> int:
        r"""Returns the number of node output features in the dataset."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        return data.y.shape[1]

    @property
    def num_edge_features(self) -> int:
        """Number of edge features"""
        if len(self) == 0:
            return 0
        data = self[0]
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            return data.edge_attr.shape[1]
        return 0

    @property
    def pos_dim(self) -> int:
        """Number of position coordinates"""
        if len(self) == 0:
            return 0
        data = self[0]
        if hasattr(data, 'pos') and data.pos is not None:
            return data.pos.shape[1]
        return 0

    def get_data_dims_dict(self) -> dict:
        r"""Returns a dictionary with the number of features for each type of data."""
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data

        edge_feature_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_feature_dim = data.edge_attr.shape[1]

        return {
            'node_feature_dim': data.x.shape[1],
            'edge_feature_dim': edge_feature_dim,
            'node_out_dim': data.y.shape[1],
            'pos_dim': data.pos.shape[1],
            'be_dim': data.boundary_encoding.shape[1]
        }


    def __len__(self):
        return self.__num_graphs
