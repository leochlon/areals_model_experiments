"""
MCMC Inference with NUTS Sampler

Markov Chain Monte Carlo inference using the No-U-Turn Sampler (NUTS) from Pyro.
Samples from the posterior distribution by evaluating the log-posterior defined by
the prior on z and the likelihood of observations given the decoder output.

Requires: pip install pyro-ppl
"""

import torch
from torch_geometric.data import Batch
from typing import Dict, Any
from pyro.infer import MCMC, NUTS


def mcmc_inference(model: torch.nn.Module, data, n_samples: int = 100,
                   warmup_steps: int = 100, sigma_tc: float = 0.01,
                   step_size: float = 0.1, max_tree_depth: int = 10,
                   print_progress: bool = False) -> Dict[str, Any]:
    """
    Performs MCMC inference using the No-U-Turn Sampler (NUTS) to sample
    from the posterior distribution of the latent variable 'z'.

    This method uses Hamiltonian Monte Carlo with automatic step size tuning
    to explore the posterior distribution defined by a Gaussian prior on z
    and a Gaussian likelihood on the observations.

    Args:
        model: Trained model with decode() method and z_dim attribute
        data: Graph with partial node features (unobserved nodes are NaN)
        n_samples: Number of posterior samples to generate
        warmup_steps: Number of warmup/burn-in steps
        sigma_tc: Standard deviation of the observation noise
        step_size: Initial step size for NUTS sampler
        max_tree_depth: Maximum tree depth for NUTS sampler
        print_progress: Whether to print progress information

    Returns:
        dict: A dictionary containing posterior samples, MAP estimate, and run statistics
    """
    model.eval()
    device = data.x.device
    z_dim = model.z_dim

    # 1. PREPARATION
    # =================
    obs_nodes_mask = ~torch.isnan(data.x).any(dim=1)
    if not obs_nodes_mask.any():
        raise ValueError("No observed (non-NaN) nodes found in data.x.")

    n_features = data.y.shape[-1]

    # Extract observed values
    y_obs = data.y[obs_nodes_mask].flatten()

    # Create data batch for decoding
    data_batch = Batch.from_data_list([data.clone()]).to(device)

    # Track MAP estimate
    max_log_posterior = -1e9
    map_z = None
    n_evaluations = 0

    # 2. DEFINE LOG-POSTERIOR
    # =========================
    def log_posterior(param):
        """
        Compute log-posterior: log p(z|y) ∝ log p(y|z) + log p(z)

        Prior: p(z) = N(0, I)
        Likelihood: p(y|z) = N(decode(z), σ²I)
        """
        nonlocal max_log_posterior, map_z, n_evaluations

        z = param['z']

        # Log-prior: -0.5 * ||z||²
        log_prior = -0.5 * torch.sum(z ** 2)

        # Decode z to get predictions
        with torch.no_grad():
            u_decoded = model.decode(z, data_batch.clone())

        # Extract predictions at observed locations
        y_pred = u_decoded[obs_nodes_mask].flatten()

        # Log-likelihood: -0.5 * ||y - y_pred||² / σ²
        residual = y_obs - y_pred
        log_likelihood = -0.5 / (sigma_tc ** 2) * torch.sum(residual ** 2)

        # Log-posterior
        log_prob = log_prior + log_likelihood

        # Track MAP estimate
        if log_prob > max_log_posterior:
            max_log_posterior = log_prob.item()
            map_z = z.clone()

        n_evaluations += 1

        return -log_prob  # NUTS minimizes the potential, so negate

    # 3. RUN MCMC SAMPLING
    # ====================
    kernel = NUTS(
        potential_fn=log_posterior,
        step_size=step_size,
        full_mass=False,
        jit_compile=True,
        max_tree_depth=max_tree_depth
    )

    mcmc = MCMC(
        kernel,
        initial_params={'z': torch.randn(1, z_dim, device=device)},
        num_samples=n_samples,
        warmup_steps=warmup_steps,
        num_chains=1,
        disable_progbar=not print_progress
    )

    if print_progress:
        print(f"Running MCMC with {n_samples} samples and {warmup_steps} warmup steps...")

    mcmc.run()

    # 4. EXTRACT RESULTS
    # ==================
    z_samples = mcmc.get_samples()['z'].reshape(-1, z_dim)

    if print_progress:
        print(f"MCMC complete. Total evaluations: {n_evaluations}")

    # 5. DECODE POSTERIOR SAMPLES
    # ============================
    u_samples_list = []

    with torch.no_grad():
        if print_progress:
            print("Decoding posterior samples...")

        for i in range(z_samples.shape[0]):
            z = z_samples[i:i+1]
            decode_batch = Batch.from_data_list([data.clone()]).to(device)
            u_decoded = model.decode(z, decode_batch)
            u_samples_list.append(u_decoded.view(data.num_nodes, n_features).cpu())

        u_posterior_samples = torch.stack(u_samples_list, dim=0)

        # Decode MAP estimate
        decode_batch_map = Batch.from_data_list([data.clone()]).to(device)
        u_map = model.decode(map_z, decode_batch_map)
        u_map = u_map.view(data.num_nodes, n_features).cpu()

    return {
        'u_posterior_samples': u_posterior_samples.squeeze(),
        'z_posterior_samples': z_samples.cpu().squeeze(),
        'u_map': u_map.squeeze(),
        'map_z': map_z.cpu().squeeze(),
        'max_log_posterior': max_log_posterior,
        'n_evaluations': n_evaluations,
        'n_samples': n_samples
    }
