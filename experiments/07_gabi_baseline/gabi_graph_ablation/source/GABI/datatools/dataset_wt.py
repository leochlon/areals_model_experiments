"""
Wind Terrain Dataset

PyTorch Geometric dataset for wind flow simulations over terrain stored in HDF5 format.
Converts 3D structured grid data into graph representations with optional height limiting
and data augmentation (cropping, flipping, rotation).
"""

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional

class WindTerrainDataset(Dataset):
    """PyTorch Geometric dataset for HDF5 wind data as grid graphs"""
    
    def __init__(self, data_path: str, mode: str, channels: List[str] = None, 
                 crop_size: Optional[tuple] = (64, 64, 64),
                 max_cells_above_terrain: Optional[int] = None,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            data_path: Path to HDF5 file
            mode: 'train', 'eval' Augmentations are applied in 'train' mode.
            channels: List of channels to use as node features (default: all available)
            crop_size: The size of the random crop (nz, ny, nx) for training.
            max_cells_above_terrain: Maximum number of cells to include above terrain surface.
                                    None means include all cells (default behavior).
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
            pre_filter: Optional pre-filter to apply to data
        """
        self.data_path = data_path
        self.mode = mode
        self.channels = channels or ['ux', 'uy', 'uz', 'p', 'turb', 'epsilon', 'nut']
        self.crop_size = crop_size
        self.max_cells_above_terrain = max_cells_above_terrain
        
        if self.mode == 'train' and self.crop_size is None:
            raise ValueError("crop_size must be specified for train mode.")
            
        # Get sample names from HDF5
        with h5py.File(data_path, 'r') as f:
            self.sample_names = list(f.keys())
        
        super().__init__(None, transform, pre_transform, pre_filter)

    def _augment_data(self, all_data: dict, terrain: np.ndarray, ds: np.ndarray) -> (dict, np.ndarray, np.ndarray):
        """
        Applies random cropping, flipping, and rotation to the data for augmentation.
        This method is only used in 'train' mode.
        """
        # --- 1. Random Cropping ---
        cz, cy, cx = self.crop_size
        sz, sy, sx = terrain.shape

        if not (cz <= sz and cy <= sy and cx <= sx):
            raise ValueError(f"Crop size {self.crop_size} is larger than data shape {terrain.shape}")
        elif (cz == sz and cy == sy and cx == sx):
            # No cropping needed
            return all_data, terrain, ds

        # Use triangular distribution for z-axis to favor crops near the ground
        start_z = int(np.random.triangular(0, 0, sz - cz))
        start_y = np.random.randint(0, sy - cy + 1)
        start_x = np.random.randint(0, sx - cx + 1)
        
        # Apply crop to all data arrays
        for channel in all_data:
            all_data[channel] = all_data[channel][start_z:start_z+cz, start_y:start_y+cy, start_x:start_x+cx]
        terrain = terrain[start_z:start_z+cz, start_y:start_y+cy, start_x:start_x+cx]

        new_ds = ds.copy()

        # --- 2. Random Flipping ---
        # Flip along X-axis
        if np.random.rand() > 0.5:
            terrain = np.flip(terrain, axis=2)
            for channel in all_data:
                all_data[channel] = np.flip(all_data[channel], axis=2)
            if 'ux' in all_data:
                all_data['ux'] *= -1.0
        
        # Flip along Y-axis
        if np.random.rand() > 0.5:
            terrain = np.flip(terrain, axis=1)
            for channel in all_data:
                all_data[channel] = np.flip(all_data[channel], axis=1)
            if 'uy' in all_data:
                all_data['uy'] *= -1.0

        # --- 3. Random Rotation (90, 180, 270 degrees) ---
        k = np.random.randint(0, 4) # Number of 90-degree CCW rotations
        if k > 0:
            # Rotate all spatial fields in the Y-X plane
            terrain = np.rot90(terrain, k=k, axes=(1, 2))
            for channel in all_data:
                all_data[channel] = np.rot90(all_data[channel], k=k, axes=(1, 2))

            # Rotate velocity components
            if 'ux' in all_data and 'uy' in all_data:
                ux_old, uy_old = all_data['ux'].copy(), all_data['uy'].copy()
                if k == 1: # 90 deg
                    all_data['ux'], all_data['uy'] = -uy_old, ux_old
                elif k == 2: # 180 deg
                    all_data['ux'], all_data['uy'] = -ux_old, -uy_old
                elif k == 3: # 270 deg
                    all_data['ux'], all_data['uy'] = uy_old, -ux_old
            
            # Swap dx and dy if rotation is 90 or 270 degrees
            if k % 2 == 1:
                new_ds[0], new_ds[1] = new_ds[1], new_ds[0]

        return all_data, terrain, new_ds
    
    @property
    def raw_file_names(self):
        return [self.data_path]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self))]
    
    def len(self):
        return len(self.sample_names)
    
    def __getitem__(self, idx):
        """Get a single graph"""
        sample_name = self.sample_names[idx]
        data = self._create_graph_data(sample_name)
        
        # Add boundary encoding
        data.boundary_encoding = self._get_boundary_encoding(data)
        
        # Create target (same as input features for consistency with HeatRectangleDataset)
        if data.y is None:
            data.y = data.x.clone()
        
        if self.transform is not None:
            data = self.transform(data)
            
        return data
    
    def _compute_terrain_surface_height(self, terrain: np.ndarray) -> np.ndarray:
        """
        Compute the terrain surface height for each (x, y) column.
        Returns a 2D array where each value is the z-index of the lowest fluid cell.
        """
        nz, ny, nx = terrain.shape
        surface_height = np.full((ny, nx), nz, dtype=int)  # Default to top if no terrain
        
        for y in range(ny):
            for x in range(nx):
                column = terrain[:, y, x]
                # Find lowest fluid cell (terrain > 0)
                fluid_indices = np.where(column > 0)[0]
                if len(fluid_indices) > 0:
                    surface_height[y, x] = fluid_indices[0]
        
        return surface_height
    
    def _apply_height_limit(self, fluid_mask: np.ndarray, terrain: np.ndarray, 
                           max_cells: int) -> np.ndarray:
        """
        Apply height limit to fluid mask, keeping only cells within max_cells above terrain.
        
        Args:
            fluid_mask: Boolean mask of fluid cells
            terrain: Terrain array (>0 for fluid)
            max_cells: Maximum cells above terrain to keep
            
        Returns:
            Modified fluid mask
        """
        if max_cells is None:
            return fluid_mask
        
        # Compute terrain surface height
        surface_height = self._compute_terrain_surface_height(terrain)
        
        # Create new mask
        nz, ny, nx = terrain.shape
        limited_mask = np.zeros_like(fluid_mask, dtype=bool)
        
        for y in range(ny):
            for x in range(nx):
                z_surface = surface_height[y, x]
                z_max = min(z_surface + max_cells, nz)
                
                # Keep cells from surface up to max height
                if z_surface < nz:
                    limited_mask[z_surface:z_max, y, x] = fluid_mask[z_surface:z_max, y, x]
        
        return limited_mask
    
    def _compute_boundary_nodes(self, fluid_indices: np.ndarray, grid_shape: tuple) -> torch.Tensor:
        """
        Compute boundary nodes from the fluid mask.
        Returns a boolean tensor where True indicates a boundary node.
        
        A fluid cell is considered a boundary node if it has at least one 
        non-fluid neighbor (including grid boundaries).
        """
        coord_to_idx = {tuple(coord): i for i, coord in enumerate(fluid_indices)}
        num_fluid_cells = len(fluid_indices)
        
        is_boundary = torch.zeros(num_fluid_cells, dtype=torch.bool)
        
        for i, (z, y, x) in enumerate(fluid_indices):
            # Check 6 neighbors (3D grid connectivity)
            neighbors = [
                (z, y, x+1), (z, y, x-1),  # x-direction
                (z, y+1, x), (z, y-1, x),  # y-direction
                (z+1, y, x), (z-1, y, x)   # z-direction
            ]
            
            for nz_n, ny_n, nx_n in neighbors:
                # Check if neighbor is outside grid bounds or not a fluid cell
                if (nz_n < 0 or nz_n >= grid_shape[0] or 
                    ny_n < 0 or ny_n >= grid_shape[1] or 
                    nx_n < 0 or nx_n >= grid_shape[2] or
                    (nz_n, ny_n, nx_n) not in coord_to_idx):
                    is_boundary[i] = True
                    break
        
        return is_boundary
    
    def _get_boundary_encoding(self, data: Data) -> torch.Tensor:
        """
        Creates a one-hot encoding for boundary nodes based on their type.
        The encoding has 3 categories:
        - [1, 0, 0]: Interior node (all 6 neighbors are fluid).
        - [0, 1, 0]: Non-slip boundary (adjacent to terrain or the bottom of the domain).
        - [0, 0, 1]: Outlet/inlet boundary (touches sides, the domain top, or the
                     artificial ceiling created by `max_cells_above_terrain`).
        """
        fluid_indices = data.fluid_indices.numpy()
        num_nodes = data.num_nodes
        coord_to_idx = {tuple(coord): i for i, coord in enumerate(fluid_indices)}
        
        # Get grid shape and the full terrain mask for boundary checks
        sz, sy, sx = data.grid_shape
        terrain_mask = data.terrain_mask

        # Tensors to mark the type of each node
        is_non_slip = torch.zeros(num_nodes, dtype=torch.bool)
        is_outlet_inlet = torch.zeros(num_nodes, dtype=torch.bool)

        # Check neighbors for each fluid node to classify it
        for i, (z, y, x) in enumerate(fluid_indices):
            # Define all 6 neighbor coordinates
            neighbors = [
                (z - 1, y, x), (z + 1, y, x),
                (z, y - 1, x), (z, y + 1, x),
                (z, y, x - 1), (z, y, x + 1)
            ]
            
            for (nz, ny, nx) in neighbors:
                # Check if the neighbor is a non-fluid cell
                if (nz, ny, nx) not in coord_to_idx:
                    
                    # Case 1: The neighbor is outside the original domain bounds.
                    if not (0 <= nz < sz and 0 <= ny < sy and 0 <= nx < sx):
                        if nz < 0:  # Bottom of the domain is a non-slip wall.
                            is_non_slip[i] = True
                        else: # Sides or top of the domain are outlet/inlet.
                            is_outlet_inlet[i] = True
                    # Case 2: The neighbor is inside the domain. We must check if it's
                    # true terrain or just an air cell excluded by the height limit.
                    else:
                        # If the terrain mask value is <= 0, it's a solid terrain cell.
                        if terrain_mask[nz, ny, nx] <= 0:
                            is_non_slip[i] = True
                        # Otherwise, it's an air cell above the height limit, which acts
                        # as a "top" outlet/inlet boundary.
                        else:
                            is_outlet_inlet[i] = True
        
        # --- Prioritization ---
        # A node adjacent to both terrain (non-slip) and an outlet wall
        # should be classified as non-slip.
        is_outlet_inlet[is_non_slip] = False

        # Interior nodes are those not classified as any kind of boundary.
        is_interior = ~(is_non_slip | is_outlet_inlet)

        # Create the final 3-dimensional one-hot encoding
        boundary_encoding = torch.zeros((num_nodes, 3), dtype=torch.float)
        boundary_encoding[is_interior, 0] = 1.0
        boundary_encoding[is_non_slip, 1] = 1.0
        boundary_encoding[is_outlet_inlet, 2] = 1.0
        
        # Sanity check: Ensure every node is assigned to exactly one category.
        assert torch.all(boundary_encoding.sum(dim=1) == 1.0), "Boundary encoding error: a node has multiple or no types."

        return boundary_encoding

    def _create_graph_data(self, sample_name: str) -> Data:
        """Create PyTorch Geometric Data object for a sample"""
        with h5py.File(self.data_path, 'r') as f:
            sample = f[sample_name]

            # Load all required channels into a dictionary of numpy arrays
            all_data = {}
            available_channels = list(sample.keys())
            for channel in self.channels:
                if channel in sample:
                    all_data[channel] = sample[channel][...]
                else:
                    print(f"  Warning: Channel '{channel}' not found in sample '{sample_name}'")
            
            if not all_data:
                raise ValueError(f"No valid channels found in sample '{sample_name}'. Available: {available_channels}")

            terrain = sample['terrain'][...]
            ds = sample['ds'][...]

            # Apply augmentation if in train mode
            if self.mode == 'train':
                all_data, terrain, ds = self._augment_data(all_data, terrain, ds)
            
            # --- IMPORTANT ---
            # Store the terrain mask *before* applying the height limit, so the
            # boundary encoding function knows what is "true" terrain.
            pre_limit_terrain_mask = torch.tensor(terrain.copy(), dtype=torch.float32)
            
            # Identify fluid cells from the (potentially augmented) terrain
            fluid_mask = terrain > 0
            
            # Apply height limit if specified
            if self.max_cells_above_terrain is not None:
                fluid_mask = self._apply_height_limit(fluid_mask, terrain, 
                                                      self.max_cells_above_terrain)
            
            grid_shape = terrain.shape
            
            # Get fluid cell indices
            fluid_indices = np.argwhere(fluid_mask)
            num_fluid_cells = len(fluid_indices)
            
            # Create node features from the (potentially augmented) data
            node_features_list = []
            for channel in self.channels:
                if channel in all_data:
                    channel_data = all_data[channel]
                    node_features_list.append(channel_data[fluid_mask])
            
            node_features = torch.tensor(np.array(node_features_list).T, dtype=torch.float32)
            
            # Create position tensor (physical coordinates) using (potentially modified) ds
            # Assumes ds is [dx, dy, dz]
            pos = torch.tensor(
                fluid_indices[:, [2, 1, 0]] * ds,
                dtype=torch.float32
            )
            
            # Create edge index using 6-connectivity
            coord_to_idx = {tuple(coord): i for i, coord in enumerate(fluid_indices)}
            edge_index = []
            
            for i, (z, y, x) in enumerate(fluid_indices):
                neighbors = [
                    (z, y, x+1), (z, y, x-1),
                    (z, y+1, x), (z, y-1, x),
                    (z+1, y, x), (z-1, y, x)
                ]
                
                for nz, ny, nx in neighbors:
                    if (nz, ny, nx) in coord_to_idx:
                        j = coord_to_idx[(nz, ny, nx)]
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                pos=pos,
                num_nodes=num_fluid_cells
            )
            
            # Add metadata
            data.sample_name = sample_name
            data.grid_shape = torch.tensor(grid_shape)
            data.ds = torch.tensor(ds, dtype=torch.float32)
            data.channel_names = self.channels
            data.terrain_mask = pre_limit_terrain_mask # Use the mask from before the height limit
            data.fluid_indices = torch.tensor(fluid_indices, dtype=torch.long)
            data.max_cells_above_terrain = self.max_cells_above_terrain
            
            return data
    
    @property
    def num_node_features(self) -> int:
        """Number of input node features"""
        if len(self) == 0: 
            return 0
        data = self[0]
        return data.x.shape[1]

    @property
    def num_node_output_features(self) -> int:
        """Number of output node features"""
        if len(self) == 0: 
            return 0
        data = self[0]
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
        """Returns a dictionary with the number of features for each type of data."""
        if len(self) == 0:
            return {'node_feature_dim': 0, 'edge_feature_dim': 0, 'node_out_dim': 0}
        
        data = self[0]
        data = data[0] if isinstance(data, tuple) else data
        
        edge_feature_dim = 0
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_feature_dim = data.edge_attr.shape[1]
        
        return {
            'node_feature_dim': data.x.shape[1], 
            'edge_feature_dim': edge_feature_dim, 
            'node_out_dim': data.y.shape[1],
            'pos_dim': data.pos.shape[1] ,
            'be_dim': data.boundary_encoding.shape[1]
        }
 