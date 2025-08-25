import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ScaleFusionUnit(nn.Module):
    """
    Fusion unit for a single scale in the BS-MSFA module.
    
    This unit fuses local and global features at a specific scale,
    with special handling for the first scale vs. deeper scales.
    """
    
    def __init__(self, local_channels: int, global_channels: int, is_first: bool = False):
        """
        Initialize the scale fusion unit.
        
        Args:
            local_channels: Number of channels in local features
            global_channels: Number of channels in global features
            is_first: Whether this is the first (shallowest) scale
        """
        super().__init__()
        self.is_first = is_first
        
        # Channel adjustment for local features
        self.local_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(local_channels, global_channels, 1),
            nn.Sigmoid()
        )
        
        # Channel adjustment for global features
        self.global_adjust = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(global_channels, local_channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        if is_first:
            # First scale fusion (no residual connection)
            self.fusion = nn.Sequential(
                nn.Conv2d(local_channels + global_channels, 
                          local_channels + global_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(local_channels + global_channels, 
                          local_channels + global_channels, 1)
            )
        else:
            # Deeper scale fusion with residual connection
            self.fusion = nn.Sequential(
                nn.Conv2d(local_channels + global_channels + (local_channels + global_channels), 
                          local_channels + global_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(local_channels + global_channels, 
                          local_channels + global_channels, 1)
            )
        
        # Channel-wise attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(local_channels + global_channels, 
                      (local_channels + global_channels) // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d((local_channels + global_channels) // 8, 
                      local_channels + global_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        local_feat: torch.Tensor, 
        global_feat: torch.Tensor,
        prev_fused: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the scale fusion unit.
        
        Args:
            local_feat: Local features tensor of shape (B, num_viewports, C_l, H, W)
            global_feat: Global features tensor of shape (B, C_g, H, W)
            prev_fused: Previous fused features (for residual connection in deeper scales)
            
        Returns:
            Fused features tensor
        """
        B, N, C_l, H, W = local_feat.shape
        _, C_g, _, _ = global_feat.shape
        
        # Process local features: average across viewports
        local_feat_avg = torch.mean(local_feat, dim=1)  # (B, C_l, H, W)
        
        # Adjust channels for fusion
        local_attn = self.local_adjust(local_feat_avg)  # (B, C_g, 1, 1)
        global_attn = self.global_adjust(global_feat)   # (B, C_l, 1, 1)
        
        # Apply attention
        local_feat_adjusted = local_feat_avg * global_attn  # (B, C_l, H, W)
        global_feat_adjusted = global_feat * local_attn     # (B, C_g, H, W)
        
        # Resize global features to match local features' spatial dimensions
        if global_feat_adjusted.shape[2:] != local_feat_adjusted.shape[2:]:
            global_feat_adjusted = F.interpolate(
                global_feat_adjusted, 
                size=local_feat_adjusted.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Concatenate features
        fused = torch.cat([local_feat_adjusted, global_feat_adjusted], dim=1)
        
        # Apply residual connection for deeper scales
        if not self.is_first and prev_fused is not None:
            # Resize previous fusion result to match current dimensions
            if prev_fused.shape[2:] != fused.shape[2:]:
                prev_fused = F.interpolate(
                    prev_fused, 
                    size=fused.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            # Concatenate with previous fusion result
            fused = torch.cat([fused, prev_fused], dim=1)
        
        # Feature fusion
        fused = self.fusion(fused)
        
        # Apply channel-wise attention
        channel_attn = self.channel_attention(fused)
        fused = fused * channel_attn
        
        return fused


class BS_MSFA(nn.Module):
    """
    Bi-Stream Multi-Scale Feature Aggregation (BS-MSFA) module.
    
    This module dynamically fuses local and global features across multiple scales
    as described in the paper, using a residual structure to achieve deeper fusion.
    """
    
    def __init__(
        self,
        local_channels: List[int],
        global_channels: List[int]
    ):
        """
        Initialize the BS-MSFA module.
        
        Args:
            local_channels: List of channel dimensions for local features at each scale
            global_channels: List of channel dimensions for global features at each scale
        """
        super().__init__()
        assert len(local_channels) == len(global_channels) == 3, \
            "BS-MSFA requires features from exactly 3 scales (S2, S3, S4)"
        
        # Create fusion units for each scale
        self.fusion_units = nn.ModuleList()
        
        # First scale fusion (shallowest - Stage 2)
        self.fusion_units.append(
            ScaleFusionUnit(
                local_channels[0], 
                global_channels[0],
                is_first=True
            )
        )
        
        # Second scale fusion (Stage 3)
        self.fusion_units.append(
            ScaleFusionUnit(
                local_channels[1],
                global_channels[1],
                is_first=False
            )
        )
        
        # Third scale fusion (deepest - Stage 4)
        self.fusion_units.append(
            ScaleFusionUnit(
                local_channels[2],
                global_channels[2],
                is_first=False
            )
        )
        
        # Final quality prediction head
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(local_channels[-1] + global_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(
        self, 
        local_features: List[torch.Tensor], 
        global_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the BS-MSFA module.
        
        Args:
            local_features: List of local feature tensors from BPR module
                           Each tensor has shape (B, num_viewports, C, H, W)
            global_features: List of global feature tensors from VMamba
                            Each tensor has shape (B, C, H, W)
                            
        Returns:
            Quality score prediction
        """
        # Initialize fusion with the first scale
        fused = self.fusion_units[0](
            local_features[0], 
            global_features[0]
        )
        
        # Hierarchical fusion through deeper scales
        for i in range(1, len(self.fusion_units)):
            fused = self.fusion_units[i](
                local_features[i], 
                global_features[i],
                fused  # Previous fusion result for residual connection
            )
        
        # Get quality score
        quality_score = self.quality_head(fused)
        
        return quality_score
    
    def get_fused_features(
        self, 
        local_features: List[torch.Tensor], 
        global_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get the fused features before the quality prediction head.
        
        Args:
            local_features: List of local feature tensors
            global_features: List of global feature tensors
            
        Returns:
            Fused features tensor
        """
        # Initialize fusion with the first scale
        fused = self.fusion_units[0](
            local_features[0], 
            global_features[0]
        )
        
        # Hierarchical fusion through deeper scales
        for i in range(1, len(self.fusion_units)):
            fused = self.fusion_units[i](
                local_features[i], 
                global_features[i],
                fused  # Previous fusion result for residual connection
            )
        
        return fused
