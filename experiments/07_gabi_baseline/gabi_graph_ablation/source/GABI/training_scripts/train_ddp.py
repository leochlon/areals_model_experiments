"""
Distributed Data Parallel Training Script

Multi-GPU training for GABI models using PyTorch DDP.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Function
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian, AddRandomWalkPE, Distance
from models import GCNGeomAutoencoder, GENGeomAutoencoder, TransformerGeomAutoencoder, MMDLoss
from datatools import (WindTerrainDataset, HeatRectangleDataset,
                       HelmholtzCarDataset, AirfoilDataset,
                       compute_dataset_stats, norm_data)
from box import Box
import yaml
from tqdm import tqdm
import string
import random
torch.multiprocessing.set_sharing_strategy('file_system')

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()
    
class GatherLayer(Function):
    """
    Gather tensors from all processes, supporting backward propagation.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def compute_loss(data, rank=None, world_size=None):
    """
    Compute loss with optional distributed MMD calculation.
    
    Args:
        data: Batch data with x (input), y (reconstruction), z (latent)
        rank: Current process rank (None for non-distributed)
        world_size: Total number of processes (None for non-distributed)
    
    Returns:
        loss, recon_loss, mmd_loss
    """
    # Compute reconstruction loss (this is fine per-GPU)
    recon_loss = torch.mean((data.x - data.y)**2.)
    
    # Get latent representations
    z_local = data.z
    
    # If not distributed or single GPU, compute normally
    if world_size is None or world_size == 1:
        mmd = MMDLoss(device=data.x.device)
        Xd = torch.randn_like(z_local).to(data.x.device)
        mmd_loss = mmd(z_local, Xd)
    else:
        # Compute MMD on gathered latents from all GPUs
        mmd_loss = compute_distributed_mmd(z_local, rank, world_size)
    
    loss = recon_loss + mmd_loss
    return loss, recon_loss, mmd_loss

def compute_distributed_mmd(z_local, rank, world_size):
    """
    Compute MMD loss across all distributed batches with proper gradient handling.
    
    Args:
        z_local: Local batch latent representations [batch_size, latent_dim]
        rank: Current process rank
        world_size: Number of processes
    
    Returns:
        mmd_loss: Scalar MMD loss computed on full distributed batch
    """
    device = z_local.device
    
    # This maintains gradients across all GPUs
    z_list = GatherLayer.apply(z_local)
    z_all = torch.cat(z_list, dim=0)

    # Generate prior samples for the full batch
    # Important: Use the same random seed across all ranks for consistency
    # Save current RNG state and set a fixed seed
    rng_state = torch.get_rng_state()
    if device.type == 'cuda':
        cuda_rng_state = torch.cuda.get_rng_state(device)
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    Xd = torch.randn_like(z_all)
    
    # Restore RNG state
    torch.set_rng_state(rng_state)
    if device.type == 'cuda':
        torch.cuda.set_rng_state(cuda_rng_state, device)
    
    # Initialize MMD loss
    mmd = MMDLoss(device=device)
    mmd_loss = mmd(z_all, Xd)
    
    # Option 2: Divide by world_size to avoid gradient accumulation
    # This is needed because each GPU computes the same loss
    mmd_loss = mmd_loss / world_size
    
    return mmd_loss

def train_distributed(rank, world_size, config_path):
    """Main training function for each process."""
    # Setup distributed training
    setup(rank, world_size)
    
    # Load the config file
    config = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)
    
    # Set up transforms
    if config.data_settings.transform == 'Cartesian':
        transform = Cartesian(norm=False)
    elif config.data_settings.transform == 'AddRandomWalkPE':
        transform = AddRandomWalkPE(walk_length=config.data_settings.random_walk_length)
    elif config.data_settings.transform == 'Distance':
        transform = Distance(norm=False)
    else:
        transform = None

    # Initialize the datasets
    dataset_type = config.data_settings.get('dataset_type', 'WindTerrain')

    # Base dataset kwargs - all datasets now use 'data_path'
    dataset_kwargs = {
        'data_path': config.io_settings.train_dataset_path,
        'transform': transform
    }
    exclude_keys = []

    # Add dataset-specific parameters
    if dataset_type == 'WindTerrain':
        dataset_class = WindTerrainDataset
        dataset_kwargs.update({
            'channels': config.data_settings.channels,
            'max_cells_above_terrain': config.data_settings.max_cells_above_terrain,
            'mode': 'train'
        })
        exclude_keys = ['terrain_mask', 'fluid_indices']
    elif dataset_type == 'HeatRectangle':
        dataset_class = HeatRectangleDataset
    elif dataset_type == 'HelmholtzCar':
        dataset_class = HelmholtzCarDataset
    elif dataset_type == 'Airfoil':
        dataset_class = AirfoilDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    train_dataset = dataset_class(**dataset_kwargs)
    
    # Create distributed sampler for data parallelism
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # Adjust batch size per GPU
    batch_size_per_gpu = config.hyperparameters.batch_size // world_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        exclude_keys=exclude_keys,
        num_workers=config.run_settings.num_t_workers,
        pin_memory=True,
        persistent_workers=False if config.run_settings.num_t_workers == 0 else True
    )

    if config.run_settings.validate:
        # Update kwargs for validation dataset
        val_kwargs = dataset_kwargs.copy()
        val_kwargs['data_path'] = config.io_settings.valid_dataset_path
        if dataset_type == 'WindTerrain':
            val_kwargs['mode'] = 'eval'

        validate_dataset = dataset_class(**val_kwargs)
        
        # Split validation indices across GPUs
        val_indices = list(range(len(validate_dataset)))
        val_indices_per_rank = val_indices[rank::world_size]
        val_subset = torch.utils.data.Subset(validate_dataset, val_indices_per_rank)
        
        validate_loader = DataLoader(
            val_subset,
            batch_size=batch_size_per_gpu,
            shuffle=False,
            num_workers=config.run_settings.num_v_workers,
            pin_memory=True,
            exclude_keys=exclude_keys,
            persistent_workers=False if config.run_settings.num_v_workers == 0 else True
        )
    
    # Get the dimensions of the data
    config.data_dims = train_dataset.get_data_dims_dict()

    # Setup run directory and naming (only on rank 0)
    if rank == 0:
        uid = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        run_name = '{}_dim_{}_uid_{}_{}gpu'.format(
            config.model_settings.model_type,
            config.model_settings.latent_dim,
            uid,
            world_size
        )
        print(f"Starting training run: {run_name}")

        # Create model saving dir and save config
        current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
        os.makedirs(os.path.join(current_run_dir, 'trained_models'), exist_ok=True)
        config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))
    else:
        run_name = None
        current_run_dir = None

    # Set device for this rank
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    
    # Initialize the model
    if config.model_settings.model_type == 'GCN':
        model = GCNGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)
    elif config.model_settings.model_type == 'GEN':
        model = GENGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)
    elif config.model_settings.model_type == 'Transformer':
        model = TransformerGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings, **config.data_settings)
    else:
        raise ValueError(f"Unknown model type: {config.model_settings.model_type}. "
                        f"Supported types: GCN, GEN, Transformer")

    # Load pretrained model if specified
    if config.io_settings.pretrained_model:
        checkpoint = torch.load(config.io_settings.pretrained_model, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Compute dataset stats (only on rank 0 and broadcast)
    if rank == 0:
        model.trainset_stats = compute_dataset_stats(train_loader, device)
    else:
        model.trainset_stats = None
    
    # Broadcast stats from rank 0 to all other ranks
    trainset_stats_list = [model.trainset_stats]
    dist.broadcast_object_list(trainset_stats_list, src=0)
    model.trainset_stats = trainset_stats_list[0]
    
    # Move model to device and wrap with DDP
    model = model.to(device)
    # Important: find_unused_parameters=True is needed when some model parameters 
    # might not be used in certain forward passes
    model = DistributedDataParallel(
        model, 
        device_ids=[rank], 
        output_device=rank,
        static_graph=True  # Set static graph for DDP
    )
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config.hyperparameters.start_lr), 
        weight_decay=float(config.hyperparameters.weight_decay)
    )
    
    # Define scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=float(config.hyperparameters.lr_decay)
    )
    
    # Training loop
    if rank == 0:
        print(f'Starting run {run_name} on {world_size} GPUs')
        print(f'Effective batch size for MMD: {config.hyperparameters.batch_size}')
        pbar = tqdm(total=config.hyperparameters.epochs)
        pbar.set_description('Training')
    
    best_validation_loss = float('inf')
    
    for epoch in range(config.hyperparameters.epochs):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        train_loss = 0
        train_recon_loss = 0
        train_mmd_loss = 0
        model.train()
        
        # Mini-batch loop
        for i_batch, data in enumerate(train_loader):
            # Norm the data
            data = norm_data(data, model.module.trainset_stats)
            
            # Move data to device
            data = data.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            data = model(data)
            
            # Compute loss with distributed MMD
            batch_loss, batch_recon_loss, batch_mmd_loss = compute_loss(data, rank, world_size)
            
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_mmd_loss += batch_mmd_loss.item()
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
        
        # Average training loss across mini-batches
        train_loss = train_loss / len(train_loader)
        train_recon_loss = train_recon_loss / len(train_loader)
        train_mmd_loss = train_mmd_loss / len(train_loader)
        
        # Reduce training loss across all ranks (for logging consistency)
        train_loss_tensor = torch.tensor([train_loss, train_recon_loss, train_mmd_loss], device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        train_loss, train_recon_loss, train_mmd_loss = train_loss_tensor.tolist()
        
        # Step scheduler
        scheduler.step()
        
        # Log training metrics (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{config.hyperparameters.epochs} - "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                  f"Loss: {train_loss:.6f}, "
                  f"Recon: {train_recon_loss:.6f}, "
                  f"MMD: {train_mmd_loss:.6f}")
        
        # Save model checkpoints (only on rank 0)
        if rank == 0 and (epoch + 1) % config.io_settings.save_epochs == 0:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'trainset_stats': model.module.trainset_stats
            }, os.path.join(current_run_dir, 'trained_models', f'e{epoch + 1}.pt'))
        
        # Validation
        if config.run_settings.validate:
            validation_loss = 0
            validation_recon_loss = 0
            validation_mmd_loss = 0
            
            model.eval()
            with torch.no_grad():
                for i_batch, data in enumerate(validate_loader):
                    # Norm the data
                    data = norm_data(data, model.module.trainset_stats)
                    
                    # Move data to device
                    data = data.to(device)
                    
                    # Forward pass
                    data = model(data)
                    
                    # Compute validation loss with distributed MMD
                    batch_loss, batch_recon_loss, batch_mmd_loss = compute_loss(data, rank, world_size)
                    validation_loss += batch_loss.item()
                    validation_recon_loss += batch_recon_loss.item()
                    validation_mmd_loss += batch_mmd_loss.item()
            
            # Average validation loss
            validation_loss = validation_loss / len(validate_loader)
            validation_recon_loss = validation_recon_loss / len(validate_loader)
            validation_mmd_loss = validation_mmd_loss / len(validate_loader)
            
            # Reduce validation loss across all ranks
            val_loss_tensor = torch.tensor([validation_loss, validation_recon_loss, validation_mmd_loss], device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            validation_loss, validation_recon_loss, validation_mmd_loss = val_loss_tensor.tolist()
            
            # Log validation metrics (only on rank 0)
            if rank == 0:
                print(f"  Validation - Loss: {validation_loss:.6f}, "
                      f"Recon: {validation_recon_loss:.6f}, "
                      f"MMD: {validation_mmd_loss:.6f}")
                
                # Update progress bar
                pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Val Loss': f'{validation_loss:.8f}'})
                pbar.update(1)
                
                # Save best model
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
                        'trainset_stats': model.module.trainset_stats
                    }, os.path.join(current_run_dir, 'trained_models', 'best.pt'))
        else:
            if rank == 0:
                pbar.set_postfix({'Train Loss': f'{train_loss:.8f}'})
                pbar.update(1)
    
    if rank == 0:
        pbar.close()
    
    cleanup()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="path to the yaml config file", type=str, required=True)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    print(f"Starting training on {world_size} GPUs")
    mp.spawn(train_distributed, args=(world_size, args.config), nprocs=world_size, join=True) 

if __name__ == "__main__":
    main()