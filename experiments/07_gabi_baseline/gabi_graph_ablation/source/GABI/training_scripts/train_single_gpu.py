"""
Single GPU Training Script

Train GABI models on a single GPU.
Supports all model types (GCN, GEN, Transformer) and datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.optim as optim
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

def compute_loss(data):
    """
    Compute loss for single GPU training.

    Args:
        data: Batch data with x (reconstruction), y (target), z (latent)

    Returns:
        loss, recon_loss, mmd_loss
    """
    # Compute reconstruction loss
    recon_loss = torch.mean((data.x - data.y)**2.)

    # Compute MMD loss
    mmd = MMDLoss(device=data.x.device)
    Xd = torch.randn_like(data.z).to(data.x.device)
    mmd_loss = mmd(data.z, Xd)

    loss = recon_loss + mmd_loss
    return loss, recon_loss, mmd_loss

def train(config_path: str):
    # load the config file
    config = Box.from_yaml(filename=parser.parse_args().config, Loader=yaml.FullLoader)
    
    if config.data_settings.transform == 'Cartesian':
        transform = Cartesian(norm=False)
    elif config.data_settings.transform == 'AddRandomWalkPE':
        transform = AddRandomWalkPE(walk_length=config.data_settings.random_walk_length)
    elif config.data_settings.transform == 'Distance':
        transform = Distance(norm=False)
    else:
        transform = None


    # initialize the datasets and dataloaders
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
    train_loader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True,
                                exclude_keys=exclude_keys,
                                num_workers=config.run_settings.num_t_workers, pin_memory=False,
                                persistent_workers=False if config.run_settings.num_t_workers == 0 else True)

    if config.run_settings.validate:
        # Update kwargs for validation dataset
        val_kwargs = dataset_kwargs.copy()
        val_kwargs['data_path'] = config.io_settings.valid_dataset_path
        if dataset_type == 'WindTerrain':
            val_kwargs['mode'] = 'eval'

        validate_dataset = dataset_class(**val_kwargs)
        validate_loader = DataLoader(validate_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False,
                                    num_workers=config.run_settings.num_v_workers, pin_memory=False, exclude_keys=exclude_keys,
                                    persistent_workers=False if config.run_settings.num_v_workers == 0 else True)
        
    # get the dimenstions of the data and add to the config dict
    config.data_dims = train_dataset.get_data_dims_dict()

    # Create run name for model saving
    uid = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    run_name = '{}_dim_{}_uid_{}'.format(config.model_settings.model_type, config.model_settings.latent_dim, uid)
    print(f"Starting training run: {run_name}")

    # Create model saving dir and save config file
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'))
    config.to_yaml(filename=os.path.join(current_run_dir, 'config.yml'))

    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('mps')
    
    # initialize the model
    if config.model_settings.model_type == 'GCN':
        model = GCNGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)
    elif config.model_settings.model_type == 'GEN':
        model = GENGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)
    elif config.model_settings.model_type == 'Transformer':
        model = TransformerGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings, **config.data_settings)
    else:
        raise ValueError(f"Unknown model type: {config.model_settings.model_type}. "
                        f"Supported types: GCN, GEN, Transformer")

    # if using a pretrained model, load it here
    if config.io_settings.pretrained_model:
        checkpoint = torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.trainset_stats  = compute_dataset_stats(train_loader, device)

    # send the models to the gpu if available
    model.to(device)

    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=float(config.hyperparameters.start_lr), weight_decay=float(config.hyperparameters.weight_decay))

    # define the learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(config.hyperparameters.lr_decay))

    # set the anomaly detection to true to interupt runs with NaNs
    # torch.autograd.set_detect_anomaly(True)

    # training loop
    print('Starting run {} on {}'.format(run_name, next(model.parameters()).device))
    pbar = tqdm(total=config.hyperparameters.epochs)
    pbar.set_description('Training')
    for epoch in range(config.hyperparameters.epochs):
        train_loss = 0
        train_recon_loss = 0
        train_mmd_loss = 0
        model.train()

        # mini-batch loop
        for i_batch, data in enumerate(train_loader):
            # norm the data
            data = norm_data(data, model.trainset_stats)
            
            # get batch data and send to the right device
            data = data.to(device)

            # reset the gradients back to zero
            optimizer.zero_grad()

            # run forward pass
            data = model(data)

            # compute the batch training loss metrics
            batch_loss, batch_recon_loss, batch_mmd_loss = compute_loss(data)
            train_loss += batch_loss.item()
            train_recon_loss += batch_recon_loss.item()
            train_mmd_loss += batch_mmd_loss.item()

            # perform SGD parameter update
            batch_loss.backward()
            optimizer.step()

        # compute the epoch training loss
        train_loss = train_loss / len(train_loader)
        train_recon_loss = train_recon_loss / len(train_loader)
        train_mmd_loss = train_mmd_loss / len(train_loader)

        # step the scheduler
        scheduler.step()

        # Log training metrics
        print(f"Epoch {epoch+1}/{config.hyperparameters.epochs} - "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, "
              f"Loss: {train_loss:.6f}, "
              f"Recon: {train_recon_loss:.6f}, "
              f"MMD: {train_mmd_loss:.6f}")

        # save the trained model every n epochs
        if (epoch + 1) % config.io_settings.save_epochs == 0:
            torch.save({'model_state_dict': model.state_dict(),'trainset_stats': model.trainset_stats},
                       os.path.join(current_run_dir, 'trained_models', 'e{}.pt'.format(epoch + 1)))


        if config.run_settings.validate:
            # compute validation loss
            validation_loss = 0
            validation_recon_loss = 0
            validation_mmd_loss = 0


            model.eval()
            with torch.no_grad():
                for i_batch, data in enumerate(validate_loader):
                    # norm the data
                    data = norm_data(data, model.trainset_stats)
                    
                    # get batch data and send to the right device, reshape globals
                    data = data.to(device)

                    # forward pass
                    data = model(data)

                    # compute the batch validation loss metrics
                    batch_loss, batch_recon_loss, batch_mmd_loss = compute_loss(data)
                    validation_loss += batch_loss.item()
                    validation_recon_loss += batch_recon_loss.item()
                    validation_mmd_loss += batch_mmd_loss.item()

            # get the full dataset validation loss for this epoch
            validation_loss =  validation_loss / len(validate_loader)
            validation_recon_loss = validation_recon_loss / len(validate_loader)
            validation_mmd_loss = validation_mmd_loss / len(validate_loader)

            # Log validation metrics
            print(f"  Validation - Loss: {validation_loss:.6f}, "
                  f"Recon: {validation_recon_loss:.6f}, "
                  f"MMD: {validation_mmd_loss:.6f}")
  
            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}','Validation Loss': f'{validation_loss:.8f}'})
            pbar.update(1)

            # save the model with the best validation loss
            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save({'model_state_dict': model.state_dict(),'trainset_stats': model.trainset_stats},
                        os.path.join(current_run_dir, 'trained_models', 'best.pt'))

        else:
            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}'})
            pbar.update(1)
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="path to the yaml config file", type=str, required=True)
    train(config_path=parser.parse_args().config)