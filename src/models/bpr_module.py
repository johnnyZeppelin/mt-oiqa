import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math
from typing import List, Tuple, Dict, Optional

class BidirectionalPseudoReference(nn.Module):
    """
    Bidirectional Pseudo-Reference (BPR) Module as described in the paper.
    
    This module:
    1. Generates pseudo-reference images from restoration and degradation directions
    2. Calculates error maps to capture quality degradation information
    3. Extracts multi-scale local features using ResNet50
    """
    
    def __init__(self, pretrained_resnet: bool = True):
        """
        Initialize the BPR module.
        
        Args:
            pretrained_resnet: Whether to use pretrained ResNet50 for feature extraction
        """
        super(BidirectionalPseudoReference, self).__init__()
        
        # Feature extraction backbone (ResNet50 as specified in the paper)
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_resnet else None
        resnet = resnet50(weights=weights)
        
        # Create a feature extraction network that returns intermediate features
        # We need features from the last three stages (stages 2, 3, and 4)
        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4
        
        # Freeze ResNet weights as per paper implementation details
        for param in self.parameters():
            param.requires_grad = False
    
    # def _calculate_error_map(self, distorted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    #     """
    #     Calculate error map between distorted and reference images with log normalization.
        
    #     As described in the paper: "error maps are calculated to capture the rich semantic 
    #     difference information from the two directions" with log normalization.
        
    #     Args:
    #         distorted: Distorted viewport image tensor
    #         reference: Reference (restored or degraded) image tensor
            
    #     Returns:
    #         Error map tensor with log normalization applied
    #     """
    #     # Calculate absolute difference
    #     error_map = torch.abs(distorted - reference)
        
    #     # Apply log normalization as mentioned in the paper
    #     # Using log(1+x) to avoid log(0) issues and provide better dynamic range
    #     error_map = torch.log1p(error_map)
        
    #     return error_map
    def _calculate_error_map(self, distorted: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Calculate error map between distorted and reference images with log normalization.
        
        As described in the paper: E = log_α(α + (I_r - I_d)^2)
        where α = ε/255^2 is a constant with ε = 0.1
        
        Args:
            distorted: Distorted viewport image tensor
            reference: Reference (restored or degraded) image tensor
            
        Returns:
            Error map tensor with log normalization applied
        """
        # Calculate squared difference
        squared_diff = (distorted - reference) ** 2
        
        # Set ε = 0.1 as specified in the paper
        epsilon = 0.1
        alpha = epsilon / (255 ** 2)
        
        # Calculate error map with log normalization
        # Using change of base formula: log_alpha(x) = ln(x) / ln(alpha)
        # Add a small epsilon to avoid log(0) issues
        epsilon_numerical = 1e-10
        error_map = torch.log(alpha + squared_diff + epsilon_numerical) / torch.log(torch.tensor(alpha) + epsilon_numerical)
        
        return error_map
    
    def _generate_restored_viewport(self, distorted: torch.Tensor) -> torch.Tensor:
        """
        Generate restored viewport using a simplified restoration approach.
        
        Note: The paper references InstructPix2Pix [83] and InstructIR [79] for restoration.
        For practical reproduction, we use a simplified approach. In a full implementation,
        this would be replaced with a pre-trained InstructPix2Pix or InstructIR model.
        
        Args:
            distorted: Distorted viewport image tensor
            
        Returns:
            Restored viewport image tensor
        """
        # In the actual paper, they use InstructPix2Pix with prompts like 
        # "Remove the distortion in the image"
        
        # For reproduction purposes, we'll use a simple identity mapping with slight modification
        # This is a placeholder that should be replaced with a proper restoration model
        restored = distorted.clone()
        
        # Simple sharpening as a minimal restoration effect
        kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=torch.float32, device=distorted.device) / 9.0
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # Apply sharpening filter (only for demonstration)
        restored = F.conv2d(
            restored, 
            kernel, 
            padding=1, 
            groups=3
        )
        
        # Clip to valid range
        restored = torch.clamp(restored, 0, 1)
        
        return restored
    
    def _generate_degraded_viewport(self, original: torch.Tensor, level: int = 3) -> torch.Tensor:
        """
        Generate degraded viewport with controlled degradation.
        
        As mentioned in the paper: "degraded viewports are mainly adding JPEG compression 
        in level 3, Gaussian noise with σ=0.008 and camera sensor noise in level 3"
        
        Args:
            original: Original viewport image tensor
            level: Degradation level (1-3)
            
        Returns:
            Degraded viewport image tensor
        """
        degraded = original.clone()
        
        # Add Gaussian noise (as mentioned in the paper)
        if level >= 1:
            noise = torch.randn_like(degraded) * 0.008  # σ=0.008 as mentioned
            degraded = degraded + noise
        
        # Apply JPEG-like compression artifacts (level 2)
        if level >= 2:
            # Simple downsampling and upsampling to simulate compression
            orig_size = degraded.shape[2:]
            degraded = F.interpolate(degraded, scale_factor=0.5, mode='bilinear', align_corners=False)
            degraded = F.interpolate(degraded, size=orig_size, mode='bilinear', align_corners=False)
        
        # Additional degradation for level 3
        if level >= 3:
            # More aggressive compression artifacts
            degraded = F.avg_pool2d(degraded, 3, stride=3)
            degraded = F.interpolate(degraded, size=orig_size, mode='bilinear', align_corners=False)
        
        # Clip to valid range
        degraded = torch.clamp(degraded, 0, 1)
        
        return degraded
    
    def forward(self, 
                viewports: torch.Tensor,
                restored_viewports: Optional[torch.Tensor] = None,
                degraded_viewports: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Process viewports through the BPR module.
        
        Args:
            viewports: Tensor of distorted viewports with shape (batch_size, num_viewports, C, H, W)
            restored_viewports: Optional pre-generated restored viewports
            degraded_viewports: Optional pre-generated degraded viewports
            
        Returns:
            List of multi-scale local features, where each element corresponds to a feature scale
            Each feature tensor has shape (batch_size, num_viewports, C, H, W)
        """
        batch_size, num_viewports, C, H, W = viewports.shape
        
        # Lists to store features at different scales
        features_s2 = []
        features_s3 = []
        features_s4 = []
        
        # Process each viewport in the batch
        for b in range(batch_size):
            for v in range(num_viewports):
                viewport = viewports[b, v].unsqueeze(0)  # Shape: (1, C, H, W)
                
                # Generate or use provided restored viewport
                if restored_viewports is not None:
                    restored = restored_viewports[b, v].unsqueeze(0)
                else:
                    restored = self._generate_restored_viewport(viewport)
                
                # Generate or use provided degraded viewport
                if degraded_viewports is not None:
                    degraded = degraded_viewports[b, v].unsqueeze(0)
                else:
                    degraded = self._generate_degraded_viewport(viewport)
                
                # Calculate error maps
                error_map_restored = self._calculate_error_map(viewport, restored)
                error_map_degraded = self._calculate_error_map(viewport, degraded)
                
                # In the paper, error maps guide the feature extraction
                # We'll concatenate the viewport with error maps for feature extraction
                input_restored = torch.cat([viewport, error_map_restored], dim=1)
                input_degraded = torch.cat([viewport, error_map_degraded], dim=1)
                
                # Extract features from both directions
                # First pass through stage1 (not used in final features but needed for subsequent stages)
                x1_restored = self.stage1(input_restored[:, :3, :, :])  # Use only RGB channels
                x1_degraded = self.stage1(input_degraded[:, :3, :, :])
                
                # Extract features from stage2
                x2_restored = self.stage2(x1_restored)
                x2_degraded = self.stage2(x1_degraded)
                
                # Extract features from stage3
                x3_restored = self.stage3(x2_restored)
                x3_degraded = self.stage3(x2_degraded)
                
                # Extract features from stage4
                x4_restored = self.stage4(x3_restored)
                x4_degraded = self.stage4(x3_degraded)
                
                # Combine features from both directions (simple concatenation)
                x2_combined = torch.cat([x2_restored, x2_degraded], dim=1)
                x3_combined = torch.cat([x3_restored, x3_degraded], dim=1)
                x4_combined = torch.cat([x4_restored, x4_degraded], dim=1)
                
                # Store features
                features_s2.append(x2_combined)
                features_s3.append(x3_combined)
                features_s4.append(x4_combined)
        
        # Reshape features to (batch_size, num_viewports, C, H, W)
        features_s2 = self._reshape_features(features_s2, batch_size, num_viewports)
        features_s3 = self._reshape_features(features_s3, batch_size, num_viewports)
        features_s4 = self._reshape_features(features_s4, batch_size, num_viewports)
        
        return [features_s2, features_s3, features_s4]
    
    def _reshape_features(self, 
                         features_list: List[torch.Tensor], 
                         batch_size: int, 
                         num_viewports: int) -> torch.Tensor:
        """
        Reshape feature list to proper batch and viewport dimensions.
        
        Args:
            features_list: List of feature tensors
            batch_size: Batch size
            num_viewports: Number of viewports per image
            
        Returns:
            Reshaped feature tensor with shape (batch_size, num_viewports, C, H, W)
        """
        # Stack all features
        features = torch.stack(features_list, dim=0)
        
        # Reshape to (batch_size, num_viewports, C, H, W)
        C, H, W = features.shape[1], features.shape[2], features.shape[3]
        features = features.view(batch_size, num_viewports, C, H, W)
        
        return features
    
    def get_feature_channels(self) -> List[int]:
        """
        Get the number of channels for each feature scale.
        
        Returns:
            List of channel dimensions for each feature scale [s2, s3, s4]
        """
        # ResNet50 channel dimensions for each stage:
        # stage2: 512 channels (256 after concatenation from both directions)
        # stage3: 1024 channels (512 after concatenation)
        # stage4: 2048 channels (1024 after concatenation)
        return [512, 1024, 2048]  # Before concatenation from both directions
        
# Add this for debugging
print(f"Error map min: {error_map.min().item()}, max: {error_map.max().item()}")