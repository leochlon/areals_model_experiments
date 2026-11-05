"""
Approximate Bayesian Computation (ABC) Inference

Quantile-based ABC for posterior inference using trained geometric autoencoders.
Generates samples from the prior, evaluates distance to observations, and accepts
the top N samples with smallest distance as the posterior.

Single GPU implementation - see abc_inference_multigpu.py for multi-GPU version.
"""

import torch
from torch_geometric.data import Batch
from typing import Dict, Any
import tqdm


def abc_inference(model: torch.nn.Module, data, n_total_samples: int = 100_000, 
                  batch_size: int = 100, n_accepted_samples: int = 1000, 
                  sigma_tc: float = 0.01, print_progress: bool = False) -> Dict[str, Any]:
    """
    Performs quantile-based Approximate Bayesian Computation (ABC) to infer
    the posterior distribution of the latent variable 'z'.

    This method generates a large number of samples from the prior, calculates
    the distance of each to the observed data, and accepts a fixed number
    of the samples with the smallest distance.

    Args:
        model: Trained model with decode() method
        data (torch_geometric.data.Data): Graph with partial node features (unobserved nodes are NaN).
        n_total_samples (int): The total number of prior samples to generate as the candidate pool.
        batch_size (int): Number of latent samples to process in parallel.
        n_accepted_samples (int): The number of top samples to accept for the posterior.
        sigma_tc (float): Standard deviation of the observation noise.
        print_progress (bool): Whether to print progress information.

    Returns:
        dict: A dictionary containing the posterior samples, best-fit result, and run statistics.
    """
    model.eval()
    device = data.x.device

    if n_accepted_samples > n_total_samples:
        raise ValueError("n_accepted_samples cannot be greater than n_total_samples.")

    # 1. PREPARATION
    # =================
    obs_nodes_mask = ~torch.isnan(data.x).any(dim=1)
    if not obs_nodes_mask.any():
        raise ValueError("No observed (non-NaN) nodes found in data.x.")

    n_obs_per_graph = int(obs_nodes_mask.sum())
    n_features = data.y.shape[-1]
    z_dim = model.z_dim

    graph_list = [data.clone() for _ in range(batch_size)]
    data_batch = Batch.from_data_list(graph_list).to(device)
    
    batch_obs_mask = ~torch.isnan(data_batch.x).any(dim=1)
    y_obs_batch = data_batch.y[batch_obs_mask].view(batch_size, n_obs_per_graph, n_features)

    # Initialize lists to store all generated samples and their norms
    all_z_priors = []
    all_norms = []
    
    # 2. MASS SAMPLING
    # ==================
    num_loops = (n_total_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        if print_progress:
            pbar = tqdm.tqdm(total=n_total_samples, desc="Sampling", unit="samples") 
        for _ in range(num_loops):
            z_prior = torch.randn(batch_size, z_dim, device=device)
            u_decoded = model.decode(z_prior, data_batch.clone())

            y_predicted = u_decoded[batch_obs_mask].view(batch_size, n_obs_per_graph, n_features)
            y_predicted_noisy = y_predicted + sigma_tc * torch.randn_like(y_predicted)
            
            diff = y_predicted_noisy - y_obs_batch
            norms = torch.norm(diff.view(batch_size, -1), p=2, dim=1)

            # Store results on CPU to conserve GPU memory
            all_z_priors.append(z_prior.cpu())
            all_norms.append(norms.cpu())
        
            if print_progress:
                pbar.update(batch_size)
        
        if print_progress:
            pbar.close()

    # 3. POSTERIOR SELECTION & DECODING
    # ==================================
    
    # Combine all results into single tensors
    all_z_priors = torch.cat(all_z_priors, dim=0)
    all_norms = torch.cat(all_norms, dim=0)

    # Sort by norm and select the top 'n_accepted_samples'
    _, sorted_indices = torch.sort(all_norms)
    top_indices = sorted_indices[:n_accepted_samples]

    # These are our posterior samples
    z_abc_posterior = all_z_priors[top_indices]

    # The "best" sample is the one with the minimum norm
    best_z = z_abc_posterior[0].unsqueeze(0).to(device)
    best_norm = all_norms[top_indices[0]].item()

    # The effective epsilon is the distance of the last accepted sample
    effective_epsilon = all_norms[top_indices[-1]].item()

    # Decode the posterior samples and the single best sample in batches
    u_abc_decode_list = []
    
    with torch.no_grad():
        if n_accepted_samples > 0:
            # Decode the single best sample first
            decode_batch_best = Batch.from_data_list([data.clone()]).to(device)
            u_decoded_best = model.decode(best_z, decode_batch_best)
            u_min_norm = u_decoded_best.view(data.num_nodes, n_features)

            # Decode all accepted posterior samples in batches
            num_decode_batches = (n_accepted_samples + batch_size - 1) // batch_size
            
            if print_progress:
                pbar_decode = tqdm.tqdm(total=n_accepted_samples, desc="Decoding posterior", unit="samples")
            
            for i in range(num_decode_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_accepted_samples)
                current_batch_size = end_idx - start_idx
                
                # Get batch of z samples and move to device
                z_batch = z_abc_posterior[start_idx:end_idx].to(device)
                
                # Create corresponding data batch
                decode_batch = Batch.from_data_list([data.clone() for _ in range(current_batch_size)]).to(device)
                
                # Decode this batch
                u_decoded = model.decode(z_batch, decode_batch)
                u_decoded_reshaped = u_decoded.view(current_batch_size, data.num_nodes, n_features)
                
                # Store results (move back to CPU to save GPU memory)
                u_abc_decode_list.append(u_decoded_reshaped.cpu())
                
                if print_progress:
                    pbar_decode.update(current_batch_size)
            
            if print_progress:
                pbar_decode.close()
            
            # Concatenate all decoded batches
            u_abc_decode = torch.cat(u_abc_decode_list, dim=0)
        else:
            u_abc_decode = torch.empty(0)
            u_min_norm = torch.empty(0)

        return {
            'u_posterior_samples': u_abc_decode.squeeze(),
            'z_posterior_samples': z_abc_posterior.squeeze(),
            'u_min_norm': u_min_norm.squeeze(),
            'best_z': best_z.squeeze(),
            'best_norm': best_norm,
            'effective_epsilon': effective_epsilon,
            'n_accepted': n_accepted_samples
        }