# File: src/examples/vmamba_demo.py

import torch
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.vmamba_module import GlobalFeatureExtractor

def main():
    # Create global feature extractor with VMamba configuration
    global_extractor = GlobalFeatureExtractor(
        in_chans=3,
        depths=[2, 2, 9, 2],  # VMamba-T configuration
        dims=[96, 192, 384, 768]
    )
    
    # Create a dummy omnidirectional image (batch_size=2, channels=3, height=512, width=1024)
    # Note: Omnidirectional images typically have a 2:1 aspect ratio (equirectangular projection)
    omnidirectional_image = torch.randn(2, 3, 512, 1024)
    
    # Extract global features
    global_features = global_extractor(omnidirectional_image)
    
    # Display feature shapes
    print("Global Feature Shapes:")
    for i, feature in enumerate(global_features):
        print(f"Stage {i+2} shape: {feature.shape} (batch_size, channels, height, width)")
    
    # Get channel dimensions
    channels = global_extractor.get_feature_channels()
    print("\nFeature channels per scale:")
    for i, channel in enumerate(channels):
        print(f"Stage {i+2}: {channel} channels")

if __name__ == "__main__":
    main()
    