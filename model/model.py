"""
This is unofficial implementation of XuNet: Structural Design of Convolutional
Neural Networks for Steganalysis . """
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor.
    
    Supports both grayscale (1 channel) and RGB (3 channels) images.
    The KV filter is applied independently to each channel.
    """

    def __init__(self, in_channels: int = 3) -> None:
        """Constructor
        
        Args:
            in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        """

        super().__init__()
        self.in_channels = in_channels
        
        # Define KV high-pass filter
        # pylint: disable=E1101
        kv = torch.tensor(
            [
                [-1.0, 2.0, -2.0, 2.0, -1.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-2.0, 8.0, -12.0, 8.0, -2.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-1.0, 2.0, -2.0, 2.0, -1.0],
            ],
            dtype=torch.float32
        ) / 12.0
        # pylint: enable=E1101
        
        # Replicate KV filter for each input channel (depthwise convolution)
        # Shape: [in_channels, 1, 5, 5]
        self.kv_filter = kv.view(1, 1, 5, 5).repeat(in_channels, 1, 1, 1)
        
        # Create depthwise convolution layer (groups=in_channels)
        self.pre_process = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=5, 
            stride=1,
            padding=2, 
            bias=False, 
            groups=in_channels
        )
        
        # Set weights to KV filter and freeze
        self.pre_process.weight.data = self.kv_filter
        self.pre_process.weight.requires_grad = False

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter
        
        Args:
            inp: Input tensor [B, C, H, W] where C is in_channels
            
        Returns:
            Filtered tensor [B, C, H, W]
        """
        return self.pre_process(inp)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        use_abs: bool = False,
    ) -> None:

        super().__init__()

        if kernel_size == 5:
            self.padding = 2
        else:
            self.padding = 0

        if activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.use_abs = use_abs
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=self.padding,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        """Forward pass following exact XuNet architecture.
        
        Order: Conv → ABS (if use_abs) → BN → Activation → Pool
        
        Args:
            inp: Input tensor
            
        Returns:
            Output tensor after conv block
        """
        # 1. Convolution
        out = self.conv(inp)
        
        # 2. Absolute value (only for Group 1)
        if self.use_abs:
            out = torch.abs(out)
        
        # 3. Batch Normalization
        out = self.batch_norm(out)
        
        # 4. Activation
        out = self.activation(out)
        
        # 5. Average Pooling
        out = self.pool(out)
        
        return out


class XuNet(nn.Module):

    def __init__(self, in_channels: int = 3) -> None:
        """Constructor
        
        Args:
            in_channels (int): Number of input channels (1=grayscale, 3=RGB). Default: 3
        """
        super().__init__()
        
        self.in_channels = in_channels
        
        # High-pass filter preprocessing (frozen, not trainable)
        self.image_processing = ImageProcessing(in_channels=in_channels)
        
        # Group 1: Conv 8×(5×5) → ABS → BN → TanH → Pool
        self.layer1 = ConvBlock(
            in_channels, 8, kernel_size=5, activation="tanh", use_abs=True
        )
        
        # Group 2: Conv 16×(5×5) → BN → TanH → Pool
        self.layer2 = ConvBlock(
            8, 16, kernel_size=5, activation="tanh", use_abs=False
        )
        
        # Group 3: Conv 32×(1×1) → BN → ReLU → Pool
        self.layer3 = ConvBlock(
            16, 32, kernel_size=1, activation="relu", use_abs=False
        )
        
        # Group 4: Conv 64×(1×1) → BN → ReLU → Pool
        self.layer4 = ConvBlock(
            32, 64, kernel_size=1, activation="relu", use_abs=False
        )
        
        # Group 5: Conv 128×(1×1) → BN → ReLU → Pool
        self.layer5 = ConvBlock(
            64, 128, kernel_size=1, activation="relu", use_abs=False
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Fully Connected Layer: 128 → 2 (as per original paper)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Forward pass through XuNet.
        
        Args:
            image: Input image tensor [B, C, H, W] where C=in_channels
            
        Returns:
            Log probabilities for 2 classes [B, 2] (cover=0, stego=1)
        """
        # High-pass filter preprocessing (frozen weights)
        with torch.no_grad():
            out = self.image_processing(image)
        
        # Group 1-5: Feature extraction
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        # Global average pooling
        out = self.gap(out)
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Classification
        out = self.fully_connected(out)
        
        return out


if __name__ == "__main__":
    # Test with RGB images (3 channels)
    print("Testing XuNet with RGB images (3 channels)...")
    net_rgb = XuNet(in_channels=3)
    print(net_rgb)
    print("\nTesting with random RGB image (512x512):")
    inp_image_rgb = torch.randn((1, 3, 512, 512))
    output_rgb = net_rgb(inp_image_rgb)
    print(f"Input shape: {inp_image_rgb.shape}")
    print(f"Output shape: {output_rgb.shape}")
    print(f"Output: {output_rgb}")
    
    # Test with grayscale images (1 channel) for backward compatibility
    print("\n" + "="*50)
    print("Testing XuNet with Grayscale images (1 channel)...")
    net_gray = XuNet(in_channels=1)
    print("\nTesting with random grayscale image (512x512):")
    inp_image_gray = torch.randn((1, 1, 512, 512))
    output_gray = net_gray(inp_image_gray)
    print(f"Input shape: {inp_image_gray.shape}")
    print(f"Output shape: {output_gray.shape}")
    print(f"Output: {output_gray}")
