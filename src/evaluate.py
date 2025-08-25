# File: src/evaluate.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.stats as stats

# Import our modules
from src.dataset import OIQADataset
from src.models.multi_task_oiqa import create_model

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Multi-task OIQA Model')
    parser.add_argument('--dataset', type=str, default='OIQA', choices=['OIQA', 'CVIQ'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing dataset')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    test_dataset = OIQADataset(
        dataset_path=os.path.join(args.data_dir, args.dataset),
        split='test',
        transform=transform
    )
    
    # Get number of classes for auxiliary tasks
    num_classes = test_dataset.get_num_classes()
    num_distortion_types = num_classes['num_distortion_types']
    num_compression_types = num_classes['num_compression_types']
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_model(
        num_distortion_types=num_distortion_types,
        num_compression_types=num_compression_types,
        dataset_name=args.dataset
    ).to(device)
    
    # Load trained model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
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
        for batch in tqdm(test_loader, desc='Evaluating'):
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
        all_predictions[k] = torch.cat(v, dim=0).cpu().numpy()
    
    for k, v in all_targets.items():
        all_targets[k] = torch.cat(v, dim=0).cpu().numpy()
    
    # Compute evaluation metrics
    quality_pred = all_predictions['quality_score'].squeeze()
    quality_true = all_targets['mos']
    
    # Pearson Linear Correlation Coefficient
    plcc = np.corrcoef(quality_pred, quality_true)[0, 1]
    
    # Spearman Rank Correlation Coefficient
    srcc = stats.spearmanr(quality_pred, quality_true).correlation
    
    # RMSE
    rmse = np.sqrt(np.mean((quality_pred - quality_true) ** 2))
    
    # Distortion classification accuracy
    distortion_pred = np.argmax(all_predictions['distortion_logits'], axis=1)
    distortion_acc = np.mean(distortion_pred == all_targets['distortion_type_idx'])
    
    # Compression classification accuracy
    compression_pred = np.argmax(all_predictions['compression_logits'], axis=1)
    compression_acc = np.mean(compression_pred == all_targets['compression_type_idx'])
    
    # Print results
    print(f"Evaluation Results on {args.dataset} dataset:")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Distortion Classification Accuracy: {distortion_acc:.4f}")
    print(f"Compression Classification Accuracy: {compression_acc:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"{args.dataset}_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"PLCC: {plcc:.4f}\n")
        f.write(f"SRCC: {srcc:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Distortion Classification Accuracy: {distortion_acc:.4f}\n")
        f.write(f"Compression Classification Accuracy: {compression_acc:.4f}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
