import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Callable
from src.config import get_config

class OIQADataset(Dataset):
    """
    Dataset class for Omnidirectional Image Quality Assessment (OIQA) tasks.
    
    This class handles loading pre-generated viewports, restored viewports, and degraded viewports
    for both CVIQ and OIQA datasets. It also loads MOS scores and auxiliary task information.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        # num_viewports: int = 20,
        # train_ratio: float = 0.8
        config: Dict = None
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the dataset root (e.g., "../data/CVIQ/" or "../data/OIQA/")
            split: 'train' or 'test' split
            transform: Optional transform to be applied to images
            num_viewports: Number of viewports per image (default: 20)
            train_ratio: Ratio of data to use for training (default: 0.8)
        """
        # Use default config if not provided
        if config is None:
            # Try to determine dataset name from path
            dataset_name = "OIQA" if "OIQA" in dataset_path else "CVIQ"
            config = get_config(dataset_name)

        self.config = config
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        # self.num_viewports = num_viewports
        # self.train_ratio = train_ratio
        self.num_viewports = config['dataset']['num_viewports']
        self.train_ratio = config['dataset']['train_ratio']
        
        # Determine dataset type from path
        self.dataset_type = "CVIQ" if "CVIQ" in dataset_path else "OIQA"
        
        # Load MOS data
        mos_path = os.path.join(dataset_path, "MOS.csv")
        if not os.path.exists(mos_path):
            raise FileNotFoundError(f"MOS.csv not found at {mos_path}")
        
        self.mos_df = pd.read_csv(mos_path)
        
        # Verify required columns exist
        required_columns = ['image_id', 'mos_score']
        if not all(col in self.mos_df.columns for col in required_columns):
            raise ValueError(f"MOS.csv must contain columns: {required_columns}")
        
        # Add auxiliary task information if available
        self.has_distortion_info = 'distortion_type' in self.mos_df.columns and 'distortion_level' in self.mos_df.columns
        self.has_compression_info = 'compression_type' in self.mos_df.columns
        
        # Create image ID list and MOS scores
        self.image_ids = self.mos_df['image_id'].tolist()
        self.mos_scores = self.mos_df['mos_score'].tolist()
        
        # Create mappings for auxiliary tasks if available
        if self.has_distortion_info:
            self.distortion_types = self.mos_df['distortion_type'].tolist()
            self.distortion_levels = self.mos_df['distortion_level'].tolist()
            
            # Create distortion type to index mapping
            self.distortion_type_to_idx = {dtype: idx for idx, dtype in enumerate(sorted(set(self.distortion_types)))}
            self.num_distortion_types = len(self.distortion_type_to_idx)
        else:
            self.distortion_types = None
            self.distortion_levels = None
            self.distortion_type_to_idx = None
            self.num_distortion_types = 0
            
        if self.has_compression_info:
            self.compression_types = self.mos_df['compression_type'].tolist()
            
            # Create compression type to index mapping
            self.compression_type_to_idx = {ctype: idx for idx, ctype in enumerate(sorted(set(self.compression_types)))}
            self.num_compression_types = len(self.compression_type_to_idx)
        else:
            self.compression_types = None
            self.compression_type_to_idx = None
            self.num_compression_types = 0
        
        # Split data into train and test
        num_samples = len(self.image_ids)
        split_idx = int(num_samples * self.train_ratio)
        
        if split == 'train':
            self.image_ids = self.image_ids[:split_idx]
            self.mos_scores = self.mos_scores[:split_idx]
            if self.has_distortion_info:
                self.distortion_types = self.distortion_types[:split_idx]
                self.distortion_levels = self.distortion_levels[:split_idx]
            if self.has_compression_info:
                self.compression_types = self.compression_types[:split_idx]
        else:
            self.image_ids = self.image_ids[split_idx:]
            self.mos_scores = self.mos_scores[split_idx:]
            if self.has_distortion_info:
                self.distortion_types = self.distortion_types[split_idx:]
                self.distortion_levels = self.distortion_levels[split_idx:]
            if self.has_compression_info:
                self.compression_types = self.compression_types[split_idx:]
        
        # Verify viewport files exist
        self._verify_viewport_files()
    
    def _verify_viewport_files(self) -> None:
        """Verify that viewport files exist for all image IDs."""
        missing_files = []
        
        for img_id in self.image_ids:
            for vp_idx in range(self.num_viewports):
                # Check original viewport
                vp_path = os.path.join(self.dataset_path, "Viewports", f"{img_id}_vp{vp_idx}.jpg")
                if not os.path.exists(vp_path):
                    # Try alternative naming conventions
                    alt_paths = [
                        os.path.join(self.dataset_path, "Viewports", f"{img_id}_viewport{vp_idx}.jpg"),
                        os.path.join(self.dataset_path, "Viewports", f"{img_id}_vp{vp_idx}.png"),
                        os.path.join(self.dataset_path, "Viewports", f"{img_id}_viewport{vp_idx}.png")
                    ]
                    found = False
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            found = True
                            break
                    if not found:
                        missing_files.append(vp_path)
                
                # Check restored viewport
                restored_vp_path = os.path.join(self.dataset_path, "RestoredViewports", f"{img_id}_vp{vp_idx}.jpg")
                if not os.path.exists(restored_vp_path):
                    # Try alternative naming conventions
                    alt_paths = [
                        os.path.join(self.dataset_path, "RestoredViewports", f"{img_id}_viewport{vp_idx}.jpg"),
                        os.path.join(self.dataset_path, "RestoredViewports", f"{img_id}_vp{vp_idx}.png"),
                        os.path.join(self.dataset_path, "RestoredViewports", f"{img_id}_viewport{vp_idx}.png")
                    ]
                    found = False
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            found = True
                            break
                    if not found:
                        missing_files.append(restored_vp_path)
                
                # Check degraded viewport
                degraded_vp_path = os.path.join(self.dataset_path, "DegradedViewports", f"{img_id}_vp{vp_idx}.jpg")
                if not os.path.exists(degraded_vp_path):
                    # Try alternative naming conventions
                    alt_paths = [
                        os.path.join(self.dataset_path, "DegradedViewports", f"{img_id}_viewport{vp_idx}.jpg"),
                        os.path.join(self.dataset_path, "DegradedViewports", f"{img_id}_vp{vp_idx}.png"),
                        os.path.join(self.dataset_path, "DegradedViewports", f"{img_id}_viewport{vp_idx}.png")
                    ]
                    found = False
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            found = True
                            break
                    if not found:
                        missing_files.append(degraded_vp_path)
        
        if missing_files:
            raise FileNotFoundError(f"Missing viewport files: {missing_files[:5]} (and {len(missing_files)-5} more)")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'viewports': Tensor of shape (num_viewports, C, H, W)
                - 'restored_viewports': Tensor of shape (num_viewports, C, H, W)
                - 'degraded_viewports': Tensor of shape (num_viewports, C, H, W)
                - 'mos': MOS score (scalar tensor)
                - 'distortion_type_idx': Distortion type index for auxiliary task 1 (scalar tensor)
                - 'distortion_level': Distortion level for auxiliary task 1 (scalar tensor)
                - 'compression_type_idx': Compression type index for auxiliary task 2 (scalar tensor)
        """
        image_id = self.image_ids[idx]
        mos = self.mos_scores[idx]
        
        # Load all viewports
        viewports = []
        restored_viewports = []
        degraded_viewports = []
        
        for vp_idx in range(self.num_viewports):
            # Find actual file paths with possible naming variations
            vp_path = self._find_viewport_path("Viewports", image_id, vp_idx)
            restored_vp_path = self._find_viewport_path("RestoredViewports", image_id, vp_idx)
            degraded_vp_path = self._find_viewport_path("DegradedViewports", image_id, vp_idx)
            
            # Load images
            viewport = Image.open(vp_path).convert('RGB')
            restored_viewport = Image.open(restored_vp_path).convert('RGB')
            degraded_viewport = Image.open(degraded_vp_path).convert('RGB')
            
            # Apply transforms if provided
            if self.transform:
                viewport = self.transform(viewport)
                restored_viewport = self.transform(restored_viewport)
                degraded_viewport = self.transform(degraded_viewport)
            else:
                # Convert to tensor if no transform is provided
                viewport = transforms.ToTensor()(viewport)
                restored_viewport = transforms.ToTensor()(restored_viewport)
                degraded_viewport = transforms.ToTensor()(degraded_viewport)
            
            viewports.append(viewport)
            restored_viewports.append(restored_viewport)
            degraded_viewports.append(degraded_viewport)
        
        # Convert lists to tensors
        viewports = torch.stack(viewports)  # Shape: (num_viewports, C, H, W)
        restored_viewports = torch.stack(restored_viewports)  # Shape: (num_viewports, C, H, W)
        degraded_viewports = torch.stack(degraded_viewports)  # Shape: (num_viewports, C, H, W)
        
        # Prepare auxiliary task data
        sample = {
            'viewports': viewports,
            'restored_viewports': restored_viewports,
            'degraded_viewports': degraded_viewports,
            'mos': torch.tensor(mos, dtype=torch.float32)
        }
        
        # Add distortion information if available
        if self.has_distortion_info:
            distortion_type = self.distortion_types[idx]
            distortion_level = self.distortion_levels[idx]
            
            # Convert distortion type to index
            distortion_type_idx = self.distortion_type_to_idx[distortion_type]
            
            sample['distortion_type_idx'] = torch.tensor(distortion_type_idx, dtype=torch.long)
            sample['distortion_level'] = torch.tensor(distortion_level, dtype=torch.float32)
        
        # Add compression information if available
        if self.has_compression_info:
            compression_type = self.compression_types[idx]
            
            # Convert compression type to index
            compression_type_idx = self.compression_type_to_idx[compression_type]
            
            sample['compression_type_idx'] = torch.tensor(compression_type_idx, dtype=torch.long)
        
        return sample
    
    def _find_viewport_path(self, folder: str, image_id: str, vp_idx: int) -> str:
        """
        Find the viewport path with possible naming variations.
        
        Args:
            folder: Folder name (Viewports, RestoredViewports, DegradedViewports)
            image_id: Image ID
            vp_idx: Viewport index
            
        Returns:
            Path to the viewport file
        """
        base_path = os.path.join(self.dataset_path, folder)
        
        # Try common naming conventions
        possible_paths = [
            os.path.join(base_path, f"{image_id}_vp{vp_idx}.jpg"),
            os.path.join(base_path, f"{image_id}_viewport{vp_idx}.jpg"),
            os.path.join(base_path, f"{image_id}_vp{vp_idx}.png"),
            os.path.join(base_path, f"{image_id}_viewport{vp_idx}.png"),
            os.path.join(base_path, f"{image_id}_VP{vp_idx}.jpg"),
            os.path.join(base_path, f"{image_id}_Viewport{vp_idx}.jpg")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no file found, raise error
        raise FileNotFoundError(f"Viewport file not found for {image_id}, viewport {vp_idx} in {folder}")
    
    def get_num_classes(self) -> Dict[str, int]:
        """
        Get number of classes for auxiliary tasks.
        
        Returns:
            Dictionary containing:
                - 'num_distortion_types': Number of distortion types
                - 'num_compression_types': Number of compression types
        """
        return {
            'num_distortion_types': self.num_distortion_types,
            'num_compression_types': self.num_compression_types
        }

if __name__ == "__main__":
    # Define standard image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create CVIQ dataset
    cviq_train = OIQADataset(
        dataset_path="../data/CVIQ/",
        split='train',
        transform=transform
    )
    
    cviq_test = OIQADataset(
        dataset_path="../data/CVIQ/",
        split='test',
        transform=transform
    )
    
    # Create OIQA dataset
    oiqqa_train = OIQADataset(
        dataset_path="../data/OIQA/",
        split='train',
        transform=transform
    )
    
    oiqqa_test = OIQADataset(
        dataset_path="../data/OIQA/",
        split='test',
        transform=transform
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    cviq_train_loader = DataLoader(
        cviq_train, 
        batch_size=16, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    # Example of getting a batch
    sample_batch = next(iter(cviq_train_loader))
    
    print("Batch shapes:")
    print(f"Viewports: {sample_batch['viewports'].shape}")
    print(f"Restored Viewports: {sample_batch['restored_viewports'].shape}")
    print(f"Degraded Viewports: {sample_batch['degraded_viewports'].shape}")
    print(f"MOS: {sample_batch['mos'].shape}")
    
    if 'distortion_type_idx' in sample_batch:
        print(f"Distortion Type Indices: {sample_batch['distortion_type_idx'].shape}")
    
    if 'compression_type_idx' in sample_batch:
        print(f"Compression Type Indices: {sample_batch['compression_type_idx'].shape}")
    
    # Get number of classes for auxiliary tasks
    num_classes = cviq_train.get_num_classes()
    print(f"Number of distortion types: {num_classes['num_distortion_types']}")
    print(f"Number of compression types: {num_classes['num_compression_types']}")
