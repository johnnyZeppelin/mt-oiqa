# File: src/models/multi_task_model.py

import torch
import torch.nn as nn
from src.models.bpr_module import BidirectionalPseudoReference
from src.models.vmamba_module import GlobalFeatureExtractor
from src.models.bs_msfa_module import BS_MSFA

class MultiTaskOIQA(nn.Module):
    """
    Complete multi-task OIQA model as described in the paper.
    
    This model integrates all components:
    - BPR module for local feature extraction
    - VMamba for global feature extraction
    - BS-MSFA for feature fusion
    - Multi-task learning heads
    """
    
    def __init__(
        self,
        num_distortion_types: int = 5,
        num_compression_types: int = 3
    ):
        """
        Initialize the multi-task OIQA model.
        
        Args:
            num_distortion_types: Number of distortion types for auxiliary task 1
            num_compression_types: Number of compression types for auxiliary task 2
        """
        super().__init__()
        
        # Feature extraction modules
        self.bpr = BidirectionalPseudoReference()
        self.global_extractor = GlobalFeatureExtractor()
        
        # Feature fusion
        self.bs_msfa = BS_MSFA(
            local_channels=self.bpr.get_feature_channels(),
            global_channels=self.global_extractor.get_feature_channels()
        )
        
        # Multi-task learning heads
        self.distortion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.bs_msfa.quality_head[2].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_distortion_types)
        )
        
        self.compression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.bs_msfa.quality_head[2].in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_compression_types)
        )
    
    def forward(
        self,
        viewports: torch.Tensor,
        restored_viewports: torch.Tensor,
        degraded_viewports: torch.Tensor,
        omnidirectional_image: torch.Tensor
    ) -> dict:
        """
        Forward pass of the multi-task OIQA model.
        
        Args:
            viewports: Distorted viewports tensor
            restored_viewports: Restored viewports tensor
            degraded_viewports: Degraded viewports tensor
            omnidirectional_image: Full omnidirectional image tensor
            
        Returns:
            Dictionary containing:
                - 'quality_score': Predicted MOS score
                - 'distortion_logits': Distortion type classification logits
                - 'compression_logits': Compression type classification logits
        """
        # Extract features
        local_features = self.bpr(
            viewports, 
            restored_viewports=restored_viewports,
            degraded_viewports=degraded_viewports
        )
        global_features = self.global_extractor(omnidirectional_image)
        
        # Fuse features
        fused_features = self.bs_msfa.get_fused_features(
            local_features, 
            global_features
        )
        
        # Main task: Quality prediction
        quality_score = self.bs_msfa.quality_head(fused_features)
        
        # Auxiliary task 1: Distortion level classification
        distortion_logits = self.distortion_head(fused_features)
        
        # Auxiliary task 2: Compression type discrimination
        compression_logits = self.compression_head(fused_features)
        
        return {
            'quality_score': quality_score,
            'distortion_logits': distortion_logits,
            'compression_logits': compression_logits
        }
    
    def compute_loss(
        self,
        predictions: dict,
        targets: dict,
        loss_weights: dict = {'quality': 1.0, 'distortion': 0.1, 'compression': 0.1}
    ) -> torch.Tensor:
        """
        Compute the total loss for the multi-task model.
        
        As described in the paper: L = Lq + 0.1Ld + 0.1Lc
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_weights: Weights for each loss component
            
        Returns:
            Total loss
        """
        # Quality prediction loss (Lq)
        quality_loss = F.mse_loss(
            predictions['quality_score'].squeeze(), 
            targets['mos']
        )
        
        # Distortion classification loss (Ld)
        distortion_loss = F.cross_entropy(
            predictions['distortion_logits'],
            targets['distortion_type_idx']
        )
        
        # Compression classification loss (Lc)
        compression_loss = F.cross_entropy(
            predictions['compression_logits'],
            targets['compression_type_idx']
        )
        
        # Weighted total loss
        total_loss = (
            loss_weights['quality'] * quality_loss +
            loss_weights['distortion'] * distortion_loss +
            loss_weights['compression'] * compression_loss
        )
        
        return total_loss
