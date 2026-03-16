# Xu-Net: Steganalysis with Deep Learning

This repository contains the implementation of Xu-Net, a convolutional neural network for steganalysis, designed to detect the presence of hidden data in digital images. The project is structured to handle dataset preparation, model training, and evaluation.

## Project Structure

```
.
├── checkpoints/        # Saved model checkpoints during training
├── dataset/            # Image datasets (cover and stego)
├── logs/               # Training logs
├── LSB/                # Scripts for LSB steganography
├── model/              # Xu-Net model definition
├── opts/               # Command-line options for scripts
├── results/            # Test results
├── utils/              # Utility functions
├── rename_dataset.py   # Script to prepare dataset filenames
├── embedding_lsb_bpp.py # Script to create stego images
├── train.py            # Main training script
├── test.py             # Script for evaluating the model
└── test_single_image.py # Script for testing a single image
```

## Workflow

The process is divided into three main stages:
1.  **Dataset Preparation**: Preparing the cover and stego images.
2.  **Training**: Training the Xu-Net model to distinguish between cover and stego images.
3.  **Testing**: Evaluating the trained model's performance.

---

### 1. Dataset Preparation

This stage involves two main steps: creating stego images from cover images and renaming the dataset files for compatibility with the data loader.

#### a. Creating Stego Images (`embedding_lsb_bpp.py`)

This script embeds a random payload into cover images using the Least Significant Bit (LSB) method to create stego images.

**Usage:**

```bash
python dataset/embedding_lsb_bpp.py [OPTIONS]
```

**Key Arguments:**

*   `--resolutions`: Specify which image resolutions to process (e.g., `res128`, `res256`). Defaults to all.
*   `--force`: Overwrite existing stego images. By default, the script will skip images that have already been processed.

**Example:**

To generate stego images for `res128` and `res256` resolutions, overwriting any existing ones:

```bash
python dataset/embedding_lsb_bpp.py --resolutions res128 res256 --force
```

#### b. Renaming Dataset Files (`rename_dataset.py`)

This script renames the image files in the `dataset` directory to include a split prefix (e.g., `train_`, `test_`, `validation_`). This is a **mandatory** step before training.

**Usage:**

The script will prompt for confirmation before renaming the files.

```bash
python rename_dataset.py
```

---

### 2. Training the Model (`train.py`)

This script trains the Xu-Net model. It supports resuming from checkpoints.

**Usage:**

```bash
python train.py [OPTIONS]
```

**Key Arguments:**

*   `--checkpoints_dir`: Directory to save model checkpoints (e.g., `./checkpoints/res128`).
*   `--cover_path`: Path to the training cover images.
*   `--stego_path`: Path to the training stego images.
*   `--valid_cover_path`: Path to the validation cover images.
*   `--valid_stego_path`: Path to the validation stego images.
*   `--train_size`: Number of training image pairs.
*   `--val_size`: Number of validation image pairs.
*   `--num_epochs`: Total number of epochs to train.
*   `--lr`: Learning rate.
*   `--batch_size`: Training batch size.
*   `--color_mode`: `RGB` for color images, `L` for grayscale.

**Example:**

To train a model on the `res128` dataset for 100 epochs:

```bash
python train.py ^
    --checkpoints_dir ./checkpoints/res128 ^
    --cover_path ./dataset/res128/train/cover ^
    --stego_path ./dataset/res128/train/stego ^
    --valid_cover_path ./dataset/res128/validation/cover ^
    --valid_stego_path ./dataset/res128/validation/stego ^
    --train_size 10000 ^
    --val_size 1000 ^
    --num_epochs 100
```

The script will automatically find the latest checkpoint in the specified directory and resume training.

---

### 3. Testing the Model

There are two ways to test the trained model.

#### a. Evaluating on a Test Set (`test.py`)

This script evaluates the model's performance on a test dataset and provides detailed metrics, including a confusion matrix.

**Usage:**

```bash
python test.py [OPTIONS]
```

**Key Arguments:**

*   `--checkpoint`: Path to the trained model checkpoint file.
*   `--test_cover_path`: Path to the test cover images.
*   `--test_stego_path`: Path to the test stego images.
*   `--test_size`: Number of test image pairs.

**Example:**

To test the model saved at epoch 100 for the `res128` resolution:

```bash
python test.py ^
    --checkpoint ./checkpoints/res128/net_100.pt ^
    --test_cover_path ./dataset/res128/test/cover ^
    --test_stego_path ./dataset/res128/test/stego ^
    --test_size 1500
```

Results are saved in the `results/{resolution}` directory.

#### b. Testing a Single Image (`test_single_image.py`)

This script allows you to quickly test a single image to determine if it is a cover or stego image.

**Usage:**

```bash
python test_single_image.py --checkpoint <path_to_checkpoint> --image_path <path_to_image>
```

**Example:**

```bash
python test_single_image.py ^
    --checkpoint ./checkpoints/res128/net_100.pt ^
    --image_path ./dataset/res128/test/stego/test_1.png
```

The script will output the prediction (Cover or Stego) along with a confidence score. Results are saved in the `results/single` directory.
