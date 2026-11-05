"""
Multi-GPU ABC Inference

Distributed ABC inference using multiple GPUs via PyTorch Geometric DataParallel.
Processes multiple prior samples in parallel across GPUs for faster posterior sampling.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import DataParallel as PyGDataParallel
from torch_geometric.data import Batch, Data
from typing import Dict, Any, List, Optional, Union
import tqdm

class PyGMethodWrapper(nn.Module):
    """
    An adapter for torch_geometric.nn.DataParallel.

    It receives a list of Data objects, which are then scattered and batched
    by PyGDataParallel. On each GPU, this wrapper's forward method receives a
    Batch object. It unpacks the 'z' attribute from it and calls the model's
    'decode' method with two arguments: z_samples and the batch object itself,
    preserving the original model's method signature.
    """
    def __init__(self, model: nn.Module, method_name: str = 'decode'):
        super().__init__()
        self.model = model
        self.method_name = method_name
        # Do not use self.__dict__['_modules']['model'] = model
        # as it can cause issues with DataParallel's replication.
        # DataParallel will handle moving self.model to devices.

    def forward(self, data_batch: Batch):
        """
        This method is called by PyGDataParallel on each target device.
        """
        if not hasattr(data_batch, 'z'):
            raise AttributeError(
                "The input Batch object must have a 'z' attribute "
                "containing the latent samples."
            )
        
        z_samples = data_batch.z
        method_to_call = getattr(self.model, self.method_name)
        return method_to_call(z_samples, data_batch)


def abc_inference(
    model: torch.nn.Module, 
    data: Data,
    n_total_samples: int = 100_000, 
    batch_size: int = 100, 
    n_accepted_samples: int = 1000, 
    sigma_tc: float = 0.01, 
    print_progress: bool = False,
    device_ids: Optional[List[int]] = None,
    observed_channels: Optional[Union[int, List[int]]] = None
) -> Dict[str, Any]:
    """
    Performs quantile-based Approximate Bayesian Computation (ABC) to infer
    the posterior distribution of the latent variable 'z' using multiple GPUs.
    
    Args:
        model: The torch.nn.Module model to use for inference
        data: The PyTorch Geometric Data object containing the graph
        n_total_samples: Total number of samples to generate
        batch_size: Batch size per GPU
        n_accepted_samples: Number of samples to accept based on lowest norms
        sigma_tc: Standard deviation for noise added to predictions
        print_progress: Whether to show progress bars
        device_ids: List of GPU device IDs to use
        observed_channels: Optional channel selection for norm computation.
                        Can be:
                        - None: use all channels (default behavior)
                        - int: use single channel at this index
                        - List[int]: use multiple channels at these indices
    """
    # 1. SETUP
    # =================
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        raise ValueError("No CUDA devices available")
    print(f"Using {len(device_ids)} GPUs: {device_ids}")
    
    primary_device = torch.device(f'cuda:{device_ids[0]}')
    
    # Move the original model to the primary device. PyGDataParallel will handle replication.
    model.to(primary_device)
    wrapped_model = PyGMethodWrapper(model, 'decode')
    model_parallel = PyGDataParallel(wrapped_model, device_ids=device_ids)
    
    # Keep the original data on the CPU for efficient cloning into lists.
    data = data.cpu()
    
    if n_accepted_samples > n_total_samples:
        raise ValueError("n_accepted_samples cannot be greater than n_total_samples.")
    
    # 2. PREPARATION
    # =================
    obs_nodes_mask = ~torch.isnan(data.x).any(dim=1)
    if not obs_nodes_mask.any():
        raise ValueError("No observed (non-NaN) nodes found in data.x.")
        
    n_obs_per_graph = int(obs_nodes_mask.sum())
    n_features = data.y.shape[-1]
    z_dim = model.z_dim
    
    # Handle channel selection
    if observed_channels is None:
        # Use all channels
        selected_channels = list(range(n_features))
    elif isinstance(observed_channels, int):
        # Single channel
        if observed_channels < 0 or observed_channels >= n_features:
            raise ValueError(f"Channel index {observed_channels} out of range [0, {n_features-1}]")
        selected_channels = [observed_channels]
    else:
        # List of channels
        selected_channels = list(observed_channels)
        for idx in selected_channels:
            if idx < 0 or idx >= n_features:
                raise ValueError(f"Channel index {idx} out of range [0, {n_features-1}]")
    
    print(f"Using channels: {selected_channels} out of {n_features} total channels for ABC norm")
    
    total_batch_size = batch_size * len(device_ids)
    
    # Select only the specified channels from the observed data
    y_obs_full = data.y[obs_nodes_mask]
    y_obs_selected = y_obs_full[:, selected_channels]
    y_obs_single = y_obs_selected.view(1, n_obs_per_graph, len(selected_channels)).to(primary_device)

    all_z_priors = []
    all_norms = []
    
    # 3. MASS SAMPLING WITH MULTI-GPU
    # ================================
    num_loops = (n_total_samples + total_batch_size - 1) // total_batch_size
    
    with torch.no_grad():
        pbar = tqdm.tqdm(total=n_total_samples, desc=f"Multi-GPU Sampling ({len(device_ids)} GPUs)", unit="samples", disable=not print_progress)
        
        for _ in range(num_loops):
            # Generate z on the primary device.
            z_prior = torch.randn(total_batch_size, z_dim, device=primary_device)
            
            # Create a LIST of Data objects on the CPU.
            graph_list_with_z = []
            for i in range(total_batch_size):
                graph_clone = data.clone()
                graph_clone.z = z_prior[i].unsqueeze(0)
                graph_list_with_z.append(graph_clone)
            
            # Pass list to PyGDataParallel. It handles scattering, batching, and moving data.
            u_decoded = model_parallel(graph_list_with_z)
            
            # The output u_decoded is gathered back on the primary_device.
            batch_obs_mask = torch.cat([obs_nodes_mask for _ in range(total_batch_size)])
            y_predicted_full = u_decoded[batch_obs_mask].view(total_batch_size, n_obs_per_graph, n_features)
            
            # Select only the specified channels for norm computation
            y_predicted_selected = y_predicted_full[:, :, selected_channels]
            y_predicted_noisy = y_predicted_selected + sigma_tc * torch.randn_like(y_predicted_selected)
            
            diff = y_predicted_noisy - y_obs_single.expand_as(y_predicted_selected)
            norms = torch.norm(diff.view(total_batch_size, -1), p=2, dim=1)
            
            all_z_priors.append(z_prior.cpu())
            all_norms.append(norms.cpu())
            
            pbar.update(min(total_batch_size, n_total_samples - pbar.n))
        pbar.close()
        
    # 4. POSTERIOR SELECTION & DECODING
    # ==================================
    all_z_priors = torch.cat(all_z_priors, dim=0)[:n_total_samples]
    all_norms = torch.cat(all_norms, dim=0)[:n_total_samples]
    
    _, sorted_indices = torch.sort(all_norms)
    top_indices = sorted_indices[:n_accepted_samples]
    
    z_abc_posterior = all_z_priors[top_indices]
    best_z_cpu = z_abc_posterior[0].unsqueeze(0)
    best_norm = all_norms[top_indices[0]].item()
    effective_epsilon = all_norms[top_indices[-1]].item()
    
    u_abc_decode_list = []
    u_min_norm = torch.empty(0)
    
    with torch.no_grad():
        if n_accepted_samples > 0:
            # Decode the single best sample first
            best_data = data.clone()
            best_data.z = best_z_cpu.to(primary_device)
            u_decoded_best = model_parallel([best_data]) # Pass as a list
            u_min_norm = u_decoded_best.view(data.num_nodes, n_features).cpu()
            
            # Decode all accepted posterior samples
            num_decode_batches = (n_accepted_samples + total_batch_size - 1) // total_batch_size
            pbar_decode = tqdm.tqdm(total=n_accepted_samples, desc=f"Multi-GPU Decoding ({len(device_ids)} GPUs)", unit="samples", disable=not print_progress)
            
            for i in range(num_decode_batches):
                start_idx = i * total_batch_size
                end_idx = min((i + 1) * total_batch_size, n_accepted_samples)
                current_batch_size = end_idx - start_idx
                
                z_batch_cpu = z_abc_posterior[start_idx:end_idx]
                z_batch = z_batch_cpu.to(primary_device)
                
                # Pad if it's the last batch
                if current_batch_size < total_batch_size:
                    padding = torch.zeros(total_batch_size - current_batch_size, z_dim, device=primary_device)
                    z_batch = torch.cat([z_batch, padding], dim=0)

                decode_graph_list = []
                for j in range(total_batch_size):
                    graph_clone = data.clone()
                    graph_clone.z = z_batch[j].unsqueeze(0)
                    decode_graph_list.append(graph_clone)
                
                u_decoded = model_parallel(decode_graph_list)
                u_decoded_reshaped = u_decoded.view(total_batch_size, data.num_nodes, n_features)
                
                u_decoded_reshaped = u_decoded_reshaped[:current_batch_size]
                u_abc_decode_list.append(u_decoded_reshaped.cpu())
                
                pbar_decode.update(current_batch_size)
            pbar_decode.close()
            
            u_abc_decode = torch.cat(u_abc_decode_list, dim=0)
        else:
            u_abc_decode = torch.empty(0)
            
    return {
        'u_posterior_samples': u_abc_decode.squeeze(),
        'z_posterior_samples': z_abc_posterior.squeeze(),
        'u_min_norm': u_min_norm.squeeze(),
        'best_z': best_z_cpu.squeeze(),
        'best_norm': best_norm,
        'effective_epsilon': effective_epsilon,
        'n_accepted': n_accepted_samples,
        'selected_channels': selected_channels
    }