"""This module is used to test the XuNet model."""
import argparse
import os
import re
import torch
import numpy as np
import imageio as io
from model.model import XuNet
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import dataset

# Test configuration
parser = argparse.ArgumentParser()
parser.add_argument("--test_cover_path", default="./dataset/res128/test/cover")
parser.add_argument("--test_stego_path", default="./dataset/res128/test/stego")
parser.add_argument("--test_size", type=int, default=1500, help="Number of test images")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--checkpoint", default="./checkpoints/net_100.pt")
parser.add_argument("--file_extension", default=".png")
parser.add_argument("--color_mode", default="RGB", choices=["RGB", "L"], 
                    help="Image color mode: 'RGB' for 3-channel or 'L' for grayscale")
args = parser.parse_args()

# Auto-detect resolution from test path (e.g., res128, res256, res512, res1024)
match = re.search(r'(res\d+)', args.test_cover_path)
resolution = match.group(1) if match else "res128"
results_dir = os.path.join("./results", resolution)
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Determine number of input channels
in_channels = 3 if args.color_mode == "RGB" else 1
print(f"Testing with {args.color_mode} images ({in_channels} channels)")
print(f"File format: {args.file_extension}")
print(f"Device: {device}")

# Load test data
test_data = dataset.DatasetLoad(
    args.test_cover_path,
    args.test_stego_path,
    args.test_size,
    transform=transforms.ToTensor(),
    file_extension=args.file_extension,
    color_mode=args.color_mode,
)

test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Load model with correct number of input channels
model = XuNet(in_channels=in_channels)
model.to(device)

# PyTorch 2.6+: Set weights_only=False for full checkpoint loading
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Testing
test_accuracy = []
all_predictions = []
all_labels = []

print("Starting testing...")

with torch.no_grad():
    for i, test_batch in enumerate(test_loader):
        images = torch.cat((test_batch["cover"], test_batch["stego"]), 0)
        labels = torch.cat((test_batch["label"][0], test_batch["label"][1]), 0)
        
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        
        outputs = model(images)
        prediction = outputs.data.max(1)[1]
        
        accuracy = (
            prediction.eq(labels.data).sum() * 100.0 / (labels.size()[0])
        )
        test_accuracy.append(accuracy.item())
        
        all_predictions.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        print(f"Batch {i+1}/{len(test_loader)}: Accuracy = {accuracy:.2f}%")

avg_accuracy = sum(test_accuracy) / len(test_accuracy)
print(f"\n{'='*50}")
print(f"Test Accuracy: {avg_accuracy:.2f}%")
print(f"{'='*50}")

# Calculate confusion matrix
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

tp = np.sum((all_predictions == 1) & (all_labels == 1))  # True Positive (Stego detected as Stego)
tn = np.sum((all_predictions == 0) & (all_labels == 0))  # True Negative (Cover detected as Cover)
fp = np.sum((all_predictions == 1) & (all_labels == 0))  # False Positive (Cover detected as Stego)
fn = np.sum((all_predictions == 0) & (all_labels == 1))  # False Negative (Stego detected as Cover)

print(f"\nConfusion Matrix:")
print(f"True Positive (Stego→Stego):   {tp}")
print(f"True Negative (Cover→Cover):   {tn}")
print(f"False Positive (Cover→Stego):  {fp}")
print(f"False Negative (Stego→Cover):  {fn}")
print(f"\nAccuracy: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")
print(f"Precision: {tp/(tp+fp)*100:.2f}%")
print(f"Recall: {tp/(tp+fn)*100:.2f}%")

# Save results to file with sequential naming (test_1, test_2, ...)
existing = [f for f in os.listdir(results_dir) if re.match(r'test_\d+\.txt', f)]
next_index = len(existing) + 1
result_file = os.path.join(results_dir, f"test_{next_index}.txt")

with open(result_file, "w") as f:
    f.write(f"{'='*50}\n")
    f.write(f"TEST RESULTS - {result_file}\n")
    f.write(f"{'='*50}\n")
    f.write(f"Resolution      : {resolution}\n")
    f.write(f"Checkpoint      : {args.checkpoint}\n")
    f.write(f"Cover path      : {args.test_cover_path}\n")
    f.write(f"Stego path      : {args.test_stego_path}\n")
    f.write(f"Test size       : {args.test_size}\n")
    f.write(f"Batch size      : {args.batch_size}\n")
    f.write(f"Color mode      : {args.color_mode}\n")
    f.write(f"Device          : {device}\n")
    f.write(f"\n{'='*50}\n")
    f.write(f"Test Accuracy   : {avg_accuracy:.2f}%\n")
    f.write(f"{'='*50}\n")
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"True Positive  (Stego->Stego) : {tp}\n")
    f.write(f"True Negative  (Cover->Cover) : {tn}\n")
    f.write(f"False Positive (Cover->Stego) : {fp}\n")
    f.write(f"False Negative (Stego->Cover) : {fn}\n")
    f.write(f"\nAccuracy  : {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%\n")
    f.write(f"Precision : {tp/(tp+fp)*100:.2f}%\n")
    f.write(f"Recall    : {tp/(tp+fn)*100:.2f}%\n")

print(f"\n✅ Results saved to: {result_file}")
