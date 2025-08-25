import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class MultiTaskOIQA(nn.Module):
    """
    Complete multi-task No-Reference Omnidirectional Image Quality Assessment model.
    
    This model integrates all components as described in the paper:
    - Bidirectional Pseudo-Reference (BPR) Module for local feature extraction
    - VMamba-based Global Feature Extractor for global feature representation
    - Bi-Stream Multi-Scale Feature Aggregation (BS-MSFA) for feature fusion
    - Multi-task learning framework with quality prediction, distortion classification, and compression type discrimination
    """
    
    def __init__(
        self,
        num_distortion_types: int,
        num_compression_types: int,
        bpr_channels: List[int] = [512, 1024, 2048],  # ResNet50 channels for stages 2, 3, 4
        vmamba_channels: List[int] = [192, 384, 768]   # VMamba channels for stages 2, 3, 4
    ):
        """
        Initialize the multi-task OIQA model.
        
        Args:
            num_distortion_types: Number of distortion types for auxiliary task 1
            num_compression_types: Number of compression types for auxiliary task 2
            bpr_channels: Channel dimensions for BPR local features at each scale
            vmamba_channels: Channel dimensions for VMamba global features at each scale
        """
        super().__init__()
        
        # Feature extraction modules
        self.bpr = BidirectionalPseudoReference()
        self.global_extractor = GlobalFeatureExtractor()
        
        # Feature fusion
        self.bs_msfa = BS_MSFA(
            local_channels=bpr_channels,
            global_channels=vmamba_channels
        )
        
        # Multi-task learning heads
        # Main task: Quality prediction
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bpr_channels[-1] + vmamba_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        # Auxiliary task 1: Distortion level classification
        self.distortion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bpr_channels[-1] + vmamba_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_distortion_types)
        )
        
        # Auxiliary task 2: Compression type discrimination
        self.compression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bpr_channels[-1] + vmamba_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_compression_types)
        )
    
    def forward(
        self,
        viewports: torch.Tensor,
        restored_viewports: torch.Tensor,
        degraded_viewports: torch.Tensor,
        omnidirectional_image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the multi-task OIQA model.
        
        Args:
            viewports: Distorted viewports tensor of shape (B, N, C, H, W)
            restored_viewports: Restored viewports tensor of shape (B, N, C, H, W)
            degraded_viewports: Degraded viewports tensor of shape (B, N, C, H, W)
            omnidirectional_image: Full omnidirectional image tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing:
                - 'quality_score': Predicted MOS score
                - 'distortion_logits': Distortion type classification logits
                - 'compression_logits': Compression type classification logits
        """
        # Extract local features from BPR module
        local_features = self.bpr(
            viewports,
            restored_viewports=restored_viewports,
            degraded_viewports=degraded_viewports
        )
        
        # Extract global features from VMamba
        global_features = self.global_extractor(omnidirectional_image)
        
        # Fuse features using BS-MSFA
        fused_features = self.bs_msfa.get_fused_features(local_features, global_features)
        
        # Main task: Quality prediction
        quality_score = self.quality_head(fused_features)
        
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
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float] = {
            'quality': 1.0,
            'distortion': 0.1,
            'compression': 0.1
        }
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
        # Using MSE loss for regression task
        quality_loss = F.mse_loss(
            predictions['quality_score'].squeeze(), 
            targets['mos']
        )
        
        # Distortion classification loss (Ld)
        # Using cross-entropy loss for classification task
        distortion_loss = F.cross_entropy(
            predictions['distortion_logits'],
            targets['distortion_type_idx']
        )
        
        # Compression classification loss (Lc)
        # Using cross-entropy loss for classification task
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
    
    def evaluate(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate model performance using standard metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing evaluation metrics:
                - 'plcc': Pearson Linear Correlation Coefficient for quality prediction
                - 'srcc': Spearman Rank Correlation Coefficient for quality prediction
                - 'rmse': Root Mean Square Error for quality prediction
                - 'distortion_acc': Accuracy for distortion classification
                - 'compression_acc': Accuracy for compression classification
        """
        results = {}
        
        # Quality prediction metrics
        quality_pred = predictions['quality_score'].squeeze().detach().cpu().numpy()
        quality_true = targets['mos'].cpu().numpy()
        
        # Pearson Linear Correlation Coefficient
        plcc = np.corrcoef(quality_pred, quality_true)[0, 1]
        results['plcc'] = plcc
        
        # Spearman Rank Correlation Coefficient
        srcc = stats.spearmanr(quality_pred, quality_true).correlation
        results['srcc'] = srcc
        
        # RMSE
        rmse = np.sqrt(np.mean((quality_pred - quality_true) ** 2))
        results['rmse'] = rmse
        
        # Distortion classification accuracy
        if 'distortion_type_idx' in targets:
            distortion_pred = torch.argmax(predictions['distortion_logits'], dim=1)
            distortion_acc = (distortion_pred == targets['distortion_type_idx']).float().mean().item()
            results['distortion_acc'] = distortion_acc
        
        # Compression classification accuracy
        if 'compression_type_idx' in targets:
            compression_pred = torch.argmax(predictions['compression_logits'], dim=1)
            compression_acc = (compression_pred == targets['compression_type_idx']).float().mean().item()
            results['compression_acc'] = compression_acc
        
        return results


def create_model(
    num_distortion_types: int,
    num_compression_types: int,
    dataset_name: str = "OIQA"
) -> MultiTaskOIQA:
    """
    Factory function to create a model with appropriate channel dimensions
    based on the dataset being used.
    
    Args:
        num_distortion_types: Number of distortion types
        num_compression_types: Number of compression types
        dataset_name: Name of the dataset (OIQA or CVIQ)
        
    Returns:
        Configured MultiTaskOIQA model
    """
    if dataset_name.lower() == "oiqa":
        # OIQA dataset specific configuration
        bpr_channels = [512, 1024, 2048]  # ResNet50 channels for stages 2, 3, 4
        vmamba_channels = [192, 384, 768]  # VMamba-T channels for stages 2, 3, 4
    elif dataset_name.lower() == "cviq":
        # CVIQ dataset specific configuration
        bpr_channels = [512, 1024, 2048]  # ResNet50 channels for stages 2, 3, 4
        vmamba_channels = [192, 384, 768]  # VMamba-T channels for stages 2, 3, 4
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return MultiTaskOIQA(
        num_distortion_types=num_distortion_types,
        num_compression_types=num_compression_types,
        bpr_channels=bpr_channels,
        vmamba_channels=vmamba_channels
    )

print(f"Local feature shapes: {[f.shape for f in local_features]}")
print(f"Global feature shapes: {[f.shape for f in global_features]}")