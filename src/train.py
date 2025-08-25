# File: src/train.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import scipy.stats as stats
import json

# Import our modules
from src.dataset import OIQADataset
from src.models.multi_task_oiqa import create_model, MultiTaskOIQA
from src.config import get_config

def parse_args():
    # parser = argparse.ArgumentParser(description='Train Multi-task OIQA Model')
    # parser.add_argument('--dataset', type=str, default='OIQA', choices=['OIQA', 'CVIQ'],
    #                     help='Dataset to use for training')
    # parser.add_argument('--data-dir', type=str, default='../data',
    #                     help='Directory containing dataset')
    # parser.add_argument('--batch-size', type=int, default=16,
    #                     help='Batch size for training')
    # parser.add_argument('--epochs', type=int, default=300,
    #                     help='Number of epochs to train')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='Learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='Momentum for SGD optimizer')
    # parser.add_argument('--weight-decay', type=float, default=1e-4,
    #                     help='Weight decay for optimizer')
    # parser.add_argument('--save-dir', type=str, default='./checkpoints',
    #                     help='Directory to save checkpoints')
    # parser.add_argument('--log-interval', type=int, default=10,
    #                     help='Interval for logging training progress')
    # parser.add_argument('--val-interval', type=int, default=1,
    #                     help='Interval for validation')
    # parser.add_argument('--resume', type=str, default=None,
    #                     help='Path to checkpoint to resume training from')
    # return parser.parse_args()
    parser = argparse.ArgumentParser(description='Train Multi-task OIQA Model')
    parser.add_argument('--dataset', type=str, default='OIQA', choices=['OIQA', 'CVIQ'],
                        help='Dataset to use for training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, loss_weights):
    model.train()
    total_loss = 0
    quality_loss = 0
    distortion_loss = 0
    compression_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        # Move data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Forward pass
        predictions = model(
            batch['viewports'],
            batch['restored_viewports'],
            batch['degraded_viewports'],
            batch['omnidirectional_image']
        )
        
        # Compute loss
        loss = model.compute_loss(predictions, batch, loss_weights)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        quality_loss += model.compute_loss(
            predictions, batch, {'quality': 1.0, 'distortion': 0, 'compression': 0}
        ).item()
        distortion_loss += model.compute_loss(
            predictions, batch, {'quality': 0, 'distortion': 1.0, 'compression': 0}
        ).item()
        compression_loss += model.compute_loss(
            predictions, batch, {'quality': 0, 'distortion': 0, 'compression': 1.0}
        ).item()
    
    # Average losses
    num_batches = len(dataloader)
    total_loss /= num_batches
    quality_loss /= num_batches
    distortion_loss /= num_batches
    compression_loss /= num_batches
    
    return {
        'total_loss': total_loss,
        'quality_loss': quality_loss,
        'distortion_loss': distortion_loss,
        'compression_loss': compression_loss
    }

def validate(model, dataloader, device):
    model.eval()
    all_predictions = {
        'quality_score': [],
        'distortion_logits': [],
        'compression_logits': []
    }
    all_targets = {
        'mos': [],
        'distortion_type_idx': [],
        'compression_type_idx': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            predictions = model(
                batch['viewports'],
                batch['restored_viewports'],
                batch['degraded_viewports'],
                batch['omnidirectional_image']
            )
            
            # Collect predictions and targets
            all_predictions['quality_score'].append(predictions['quality_score'])
            all_predictions['distortion_logits'].append(predictions['distortion_logits'])
            all_predictions['compression_logits'].append(predictions['compression_logits'])
            
            all_targets['mos'].append(batch['mos'])
            all_targets['distortion_type_idx'].append(batch['distortion_type_idx'])
            all_targets['compression_type_idx'].append(batch['compression_type_idx'])
    
    # Concatenate all predictions and targets
    for k, v in all_predictions.items():
        all_predictions[k] = torch.cat(v, dim=0)
    
    for k, v in all_targets.items():
        all_targets[k] = torch.cat(v, dim=0)
    
    # Compute evaluation metrics
    metrics = model.evaluate(all_predictions, all_targets)
    
    return metrics

# def main():
#     args = parse_args()
    
#     # Set up device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Create save directory
#     os.makedirs(args.save_dir, exist_ok=True)
    
#     # Define transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     # Create datasets
#     train_dataset = OIQADataset(
#         dataset_path=os.path.join(args.data_dir, args.dataset),
#         split='train',
#         transform=transform
#     )
    
#     val_dataset = OIQADataset(
#         dataset_path=os.path.join(args.data_dir, args.dataset),
#         split='test',
#         transform=transform
#     )
    
#     # Get number of classes for auxiliary tasks
#     num_classes = train_dataset.get_num_classes()
#     num_distortion_types = num_classes['num_distortion_types']
#     num_compression_types = num_classes['num_compression_types']
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Create model
#     model = create_model(
#         num_distortion_types=num_distortion_types,
#         num_compression_types=num_compression_types,
#         dataset_name=args.dataset
#     ).to(device)
    
#     # Create optimizer
#     optimizer = optim.SGD(
#         model.parameters(),
#         lr=args.lr,
#         momentum=args.momentum,
#         weight_decay=args.weight_decay
#     )
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
#     # Loss weights as specified in the paper
#     loss_weights = {
#         'quality': 1.0,
#         'distortion': 0.1,
#         'compression': 0.1
#     }
    
#     # Resume from checkpoint if specified
#     start_epoch = 0
#     if args.resume:
#         checkpoint = torch.load(args.resume)
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
#         start_epoch = checkpoint['epoch'] + 1
#         print(f"Resumed training from epoch {start_epoch}")
    
#     # Training loop
#     best_srcc = 0.0
#     for epoch in range(start_epoch, args.epochs):
#         print(f"Epoch {epoch+1}/{args.epochs}")
        
#         # Train for one epoch
#         train_results = train_epoch(
#             model, train_loader, optimizer, device, loss_weights
#         )
        
#         # Print training results
#         print(f"Train Loss: {train_results['total_loss']:.4f} "
#               f"(Quality: {train_results['quality_loss']:.4f}, "
#               f"Distortion: {train_results['distortion_loss']:.4f}, "
#               f"Compression: {train_results['compression_loss']:.4f})")
        
#         # Validate if needed
#         if (epoch + 1) % args.val_interval == 0:
#             val_metrics = validate(model, val_loader, device)
            
#             # Print validation metrics
#             print(f"Validation Metrics: "
#                   f"PLCC: {val_metrics['plcc']:.4f}, "
#                   f"SRCC: {val_metrics['srcc']:.4f}, "
#                   f"RMSE: {val_metrics['rmse']:.4f}")
            
#             # Save best model
#             if val_metrics['srcc'] > best_srcc:
#                 best_srcc = val_metrics['srcc']
#                 checkpoint_path = os.path.join(
#                     args.save_dir, 
#                     f"{args.dataset}_best_srcc.pth"
#                 )
#                 torch.save({
#                     'model': model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'scheduler': scheduler.state_dict(),
#                     'epoch': epoch,
#                     'srcc': best_srcc
#                 }, checkpoint_path)
#                 print(f"Saved best model with SRCC: {best_srcc:.4f}")
        
#         # Update learning rate
#         scheduler.step()
        
#         # Save checkpoint
#         if (epoch + 1) % 10 == 0:
#             checkpoint_path = os.path.join(
#                 args.save_dir, 
#                 f"{args.dataset}_epoch_{epoch+1}.pth"
#             )
#             torch.save({
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'scheduler': scheduler.state_dict(),
#                 'epoch': epoch,
#                 'srcc': val_metrics['srcc'] if 'val_metrics' in locals() else 0.0
#             }, checkpoint_path)
    
#     print("Training completed!")
def main():
    args = parse_args()
    
    # Get configuration
    config = get_config(args.dataset)
    
    # Override with custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            # Merge custom config with default config
            for k, v in custom_config.items():
                if k in config:
                    config[k].update(v)
                else:
                    config[k] = v
    
    # Create directories
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['results'], exist_ok=True)
    
    # Save config for reproducibility
    config_path = os.path.join(config['paths']['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set up device
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(config['dataset']['image_size']['viewport']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = OIQADataset(
        dataset_path=os.path.join(config['dataset']['root'], config['dataset']['name']),
        split='train',
        transform=transform,
        train_ratio=config['dataset']['train_ratio']
    )
    
    val_dataset = OIQADataset(
        dataset_path=os.path.join(config['dataset']['root'], config['dataset']['name']),
        split='test',
        transform=transform,
        train_ratio=config['dataset']['train_ratio']
    )
    
    # Get number of classes for auxiliary tasks
    num_classes = train_dataset.get_num_classes()
    num_distortion_types = num_classes['num_distortion_types']
    num_compression_types = num_classes['num_compression_types']
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create model
    model = create_model(
        num_distortion_types=num_distortion_types,
        num_compression_types=num_compression_types,
        dataset_name=config['dataset']['name']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_scheduler']['step_size'],
        gamma=config['training']['lr_scheduler']['gamma']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    best_srcc = 0.0
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_results = train_epoch(
            model, train_loader, optimizer, device, loss_weights
        )
        
        # Print training results
        print(f"Train Loss: {train_results['total_loss']:.4f} "
              f"(Quality: {train_results['quality_loss']:.4f}, "
              f"Distortion: {train_results['distortion_loss']:.4f}, "
              f"Compression: {train_results['compression_loss']:.4f})")
        
        # Validate if needed
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device)
            
            # Print validation metrics
            print(f"Validation Metrics: "
                  f"PLCC: {val_metrics['plcc']:.4f}, "
                  f"SRCC: {val_metrics['srcc']:.4f}, "
                  f"RMSE: {val_metrics['rmse']:.4f}")
            
            # Save best model
            if val_metrics['srcc'] > best_srcc:
                best_srcc = val_metrics['srcc']
                checkpoint_path = os.path.join(
                    args.save_dir, 
                    f"{args.dataset}_best_srcc.pth"
                )
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'srcc': best_srcc
                }, checkpoint_path)
                print(f"Saved best model with SRCC: {best_srcc:.4f}")
        
        # Update learning rate
        scheduler.step()
        # Added
        print(f"Loss components - Quality: {quality_loss:.4f}, Distortion: {distortion_loss:.4f}, Compression: {compression_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f"{args.dataset}_epoch_{epoch+1}.pth"
            )
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'srcc': val_metrics['srcc'] if 'val_metrics' in locals() else 0.0
            }, checkpoint_path)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
