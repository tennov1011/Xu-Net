"""This module provides method to enter various input to the model training."""
import argparse


def arguments() -> str:
    """This function returns arguments."""

    parser = argparse.ArgumentParser()
    
    # Dataset paths - Update sesuai lokasi dataset Anda
    parser.add_argument(
        "--cover_path",
        default="./dataset/res128/train/cover",
        help="Path to training cover images"
    )
    parser.add_argument(
        "--stego_path",
        default="./dataset/res128/train/stego",
        help="Path to training stego images"
    )
    parser.add_argument(
        "--valid_cover_path",
        default="./dataset/res128/validation/cover",
        help="Path to validation cover images"
    )
    parser.add_argument(
        "--valid_stego_path",
        default="./dataset/res128/validation/stego",
        help="Path to validation stego images"
    )
    parser.add_argument(
        "--test_cover_path",
        default="./dataset/res128/test/stego",
        help="Path to test stego images"
    )
    parser.add_argument(
        "--test_stego_path",
        default="./dataset/res128/test/stego",
        help="Path to test stego images"
    )
    
    # File extension (pgm or png)
    parser.add_argument(
        "--file_extension",
        default=".png",
        help="Image file extension (.pgm or .png)"
    )
    
    # Inference
    parser.add_argument('--test_image', type=str, default=None,
                    help='Path to test image')
    parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to model checkpoint')
    parser.add_argument('--resolution', type=str, default='128x128',
                    choices=['128x128', '256x256', '512x512'],
                    help='Image resolution')
    
    # Color mode (RGB or grayscale)
    parser.add_argument(
        "--color_mode",
        default="RGB",
        choices=["RGB", "L"],
        help="Image color mode: 'RGB' for 3-channel or 'L' for grayscale (1-channel)"
    )
    

    # Training parameters
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--train_size", type=int, default=7000, help="Number of training image pairs")
    parser.add_argument("--val_size", type=int, default=1500, help="Number of validation image pairs")
    parser.add_argument("--test_size", type=int, default=1500, help="Number of test image pairs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")


    opt = parser.parse_args()
    return opt
