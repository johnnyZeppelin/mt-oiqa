# File: src/examples/bs_msfa_demo.py

import torch
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.bpr_module import BidirectionalPseudoReference
from src.models.vmamba_module import GlobalFeatureExtractor
from src.models.bs_msfa_module import BS_MSFA

def main():
    # Set up paths
    data_root = os.path.join(project_root, "data")
    
    # Create modules
    bpr = BidirectionalPseudoReference()
    global_extractor = GlobalFeatureExtractor()
    bs_msfa = BS_MSFA(
        local_channels=bpr.get_feature_channels(),
        global_channels=global_extractor.get_feature_channels()
    )
    
    # Create dummy data
    # Viewports: (batch_size, num_viewports, C, H, W)
    viewports = torch.randn(4, 20, 3, 224, 224)
    restored_viewports = torch.randn(4, 20, 3, 224, 224)
    degraded_viewports = torch.randn(4, 20, 3, 224, 224)
    
    # Full omnidirectional image: (batch_size, C, H, W)
    omnidirectional_image = torch.randn(4, 3, 512, 1024)  # 2:1 aspect ratio
    
    # Extract features
    local_features = bpr(
        viewports, 
        restored_viewports=restored_viewports,
        degraded_viewports=degraded_viewports
    )
    global_features = global_extractor(omnidirectional_image)
    
    # Fuse features and get quality prediction
    quality_score = bs_msfa(local_features, global_features)
    
    print(f"Quality score shape: {quality_score.shape} (batch_size, 1)")
    print(f"Sample quality scores: {quality_score.squeeze().detach().numpy()}")

if __name__ == "__main__":
    main()