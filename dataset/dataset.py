"""This module provide the data sample for training."""

import os
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image

import imageio.v2 as io # Menggunakan v2 untuk menghindari warning
# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101


class DatasetLoad(Dataset):
    """Dataset loader for training/validation/testing.
    
    Supports both grayscale and RGB images in PNG or PGM format.
    Images are automatically converted to the appropriate format based on their content.
    Reads filenames dynamically to prevent hardcoded index errors.
    """

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int = None,
        transform: Tuple = None,
        file_extension: str = ".png",
        color_mode: str = "RGB",
    ) -> None:
        """Constructor."""
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform
        self.file_extension = file_extension
        self.color_mode = color_mode  # 'RGB' or 'L' (grayscale)

        # --- Membaca nama file secara dinamis ---
        # Mengambil semua file di folder cover yang sesuai ekstensi
        all_files = sorted([
            f for f in os.listdir(cover_path) 
            if f.lower().endswith(file_extension.lower())
        ])
        
        # Jika parameter size diisi (misal untuk membatasi dataset), potong list-nya.
        # Jika tidak diisi atau lebih besar dari jumlah file, gunakan semua file yang ada.
        if size is not None and 0 < size < len(all_files):
            self.image_filenames = all_files[:size]
        else:
            self.image_filenames = all_files
            
        self.data_size = len(self.image_filenames)

    def __len__(self) -> int:
        """returns the length of the dataset."""
        return self.data_size

    def __getitem__(self, index: int) -> dict:
        """Returns item at index."""
        
        # ---Menggunakan nama file asli dari list ---
        img_name = self.image_filenames[index]
        
        cover_path = os.path.join(self.cover, img_name)
        stego_path = os.path.join(self.stego, img_name)
        
        # Load images using PIL for better PNG support
        try:
            cover_img = Image.open(cover_path)
            stego_img = Image.open(stego_path)
            
            # Convert to specified color mode (RGB or L)
            cover_img = cover_img.convert(self.color_mode)
            stego_img = stego_img.convert(self.color_mode)
            
        except Exception as e:
            # Fallback to imageio if PIL fails
            print(f"Warning: PIL failed for {img_name}, using imageio. Error: {e}")
            cover_img = io.imread(cover_path)
            stego_img = io.imread(stego_path)
            
            # Ensure proper shape for RGB
            if self.color_mode == "RGB":
                if len(cover_img.shape) == 2:  # Grayscale
                    cover_img = np.stack([cover_img] * 3, axis=-1)
                if len(stego_img.shape) == 2:
                    stego_img = np.stack([stego_img] * 3, axis=-1)
            else:  # Grayscale
                if len(cover_img.shape) == 3:  # RGB
                    cover_img = np.mean(cover_img, axis=-1).astype(np.uint8)
                if len(stego_img.shape) == 3:
                    stego_img = np.mean(stego_img, axis=-1).astype(np.uint8)
        
        # Create labels
        # pylint: disable=E1101
        label1 = torch.tensor(0, dtype=torch.long).to(device)  # Cover label
        label2 = torch.tensor(1, dtype=torch.long).to(device)  # Stego label
        # pylint: enable=E1101
        
        # Apply transforms if specified
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
        
        sample = {
            "cover": cover_img, 
            "stego": stego_img,
            "label": [label1, label2]
        }
        
        return sample