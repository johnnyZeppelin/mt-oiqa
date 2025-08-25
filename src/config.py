# File: src/config.py

import os
from pathlib import Path

def get_config(dataset_name="OIQA"):
    """
    Get configuration for the OIQA model.
    
    Args:
        dataset_name: Name of the dataset (OIQA or CVIQ)
        
    Returns:
        Dictionary containing configuration parameters
    """
    # Project root directory
    project_root = str(Path(__file__).parent.parent)
    
    # Base configuration
    config = {
        # Dataset configuration
        'dataset': {
            'name': dataset_name,
            'root': os.path.join(project_root, 'data'),
            'train_ratio': 0.8,
            'num_viewports': 20,
            'image_size': {
                'viewport': (224, 224),
                'omnidirectional': (512, 1024)  # 2:1 aspect ratio for equirectangular projection
            }
        },
        
        # Model configuration
        'model': {
            'bpr': {
                'pretrained_resnet': True,
                'feature_stages': [2, 3, 4]  # Last three stages of ResNet50
            },
            'vmamba': {
                'type': 'VMamba-T',  # Tiny version as used in the paper
                'depths': [2, 2, 9, 2],
                'dims': [96, 192, 384, 768],
                'd_state': 16,
                'drop_path_rate': 0.2
            },
            'bs_msfa': {
                'feature_stages': [1, 2, 3]  # Last three stages (0-indexed)
            },
            'multi_task': {
                'loss_weights': {
                    'quality': 1.0,
                    'distortion': 0.1,
                    'compression': 0.1
                }
            }
        },
        
        # Training configuration
        'training': {
            'optimizer': 'SGD',
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'batch_size': 16,
            'epochs': 300,
            'lr_scheduler': {
                'type': 'StepLR',
                'step_size': 1,
                'gamma': 0.9
            },
            'num_workers': 4,
            'pin_memory': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        
        # Evaluation configuration
        'evaluation': {
            'metrics': ['PLCC', 'SRCC', 'RMSE']
        },
        
        # Paths
        'paths': {
            'checkpoints': os.path.join(project_root, 'checkpoints', dataset_name),
            'logs': os.path.join(project_root, 'logs', dataset_name),
            'results': os.path.join(project_root, 'results', dataset_name)
        }
    }

    # Dataset-specific configuration
    if dataset_name == "OIQA":
        config['dataset'].update({
            'distortion_types': ['JPEG', 'JP2K', 'GN', 'GB'],
            'compression_types': ['H.264/AVC', 'H.265/HEVC']
        })
    elif dataset_name == "CVIQ":
        config['dataset'].update({
            'distortion_types': ['JPEG', 'H.264/AVC', 'H.265/HEVC'],
            'compression_types': ['H.264/AVC', 'H.265/HEVC']
        })
    
    return config