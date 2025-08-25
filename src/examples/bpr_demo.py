# File: src/examples/bpr_demo.py

import torch
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import OIQADataset
from src.models.bpr_module import BidirectionalPseudoReference
from torchvision import transforms

def main():
    # Define the correct relative path to datasets
    # If this script is in src/examples/, then data is at ../../data/
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", "data")
    
    # Create dataset
    dataset = OIQADataset(
        dataset_path=os.path.join(data_root, "CVIQ"),
        split='train',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    
    # Get a batch of data
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    
    # Initialize BPR module
    bpr = BidirectionalPseudoReference(pretrained_resnet=True)
    
    # Process through BPR module
    local_features = bpr(
        batch['viewports'],
        restored_viewports=batch['restored_viewports'],
        degraded_viewports=batch['degraded_viewports']
    )
    
    # Display feature shapes
    print("BPR Module Output Shapes:")
    for i, feature in enumerate(local_features):
        print(f"Feature scale {i+2} shape: {feature.shape} "
              f"(batch_size, num_viewports, channels, height, width)")

if __name__ == "__main__":
    main()
