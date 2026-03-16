"""Test single stego image using trained XuNet model."""

import argparse
import os
import re
import torch
from PIL import Image
import torchvision.transforms as transforms
from model.model import XuNet

def parse_args():
    parser = argparse.ArgumentParser(description="Test single image")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (e.g., ./checkpoints/res128/net_100.pt)")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to image to test")
    parser.add_argument("--color_mode", type=str, default="RGB",
                        choices=["RGB", "L"], help="Color mode (RGB or L for grayscale)")
    return parser.parse_args()

def load_model(checkpoint_path, in_channels):
    """Load trained model from checkpoint"""
    model = XuNet(in_channels=in_channels)
    
    # PyTorch 2.6+: Set weights_only=False for full checkpoint loading
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('epoch', 'unknown')

def predict_image(model, image_path, color_mode, device):
    """Predict if image is stego or cover"""
    # Load image
    image = Image.open(image_path).convert(color_mode)
    
    # Get image info
    img_width, img_height = image.size
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    return predicted_class, confidence, probabilities[0].cpu().numpy(), (img_width, img_height)

def main():
    args = parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = 3 if args.color_mode == "RGB" else 1
    
    print("="*60)
    print("SINGLE IMAGE STEGANALYSIS TEST")
    print("="*60)
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Test image: {args.image_path}")
    print(f"Color mode: {args.color_mode}")
    print(f"Device: {device}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, epoch = load_model(args.checkpoint, in_channels)
    model = model.to(device)
    print(f"✅ Model loaded successfully! (Trained epoch: {epoch})")
    
    # Predict
    print("\nAnalyzing image...")
    predicted_class, confidence, probs, img_size = predict_image(
        model, args.image_path, args.color_mode, device
    )
    
    
    class_names = ["Cover", "Stego"]

    print(f"\nProbability Distribution:")
    print(f"  Cover: {probs[0]*100:.2f}%")
    print(f"  Stego: {probs[1]*100:.2f}%")
    
    print("\n" + "-"*60)
    if predicted_class == 1:
        print(f"⚠️  IMAGE IS STEGO! (Confidence: {confidence:.2f}%)")
        if confidence >= 90:
            print("   Strong detection - High confidence")
        elif confidence >= 70:
            print("   Moderate detection - Medium confidence")
        else:
            print("   Weak detection - Low confidence")
    else:
        print(f"✅ IMAGE IS COVER (Confidence: {confidence:.2f}%)")
        if confidence >= 90:
            print("   Strong confidence - Likely clean image")
        elif confidence >= 70:
            print("   Moderate confidence")
        else:
            print("   Weak confidence - May contain hidden data")
    print("-"*60)

    # Save results to results/single/ with sequential naming
    results_dir = os.path.join("./results", "single")
    os.makedirs(results_dir, exist_ok=True)
    existing = [f for f in os.listdir(results_dir) if re.match(r'test_\d+\.txt', f)]
    next_index = len(existing) + 1
    result_file = os.path.join(results_dir, f"test_{next_index}.txt")

    with open(result_file, "w") as f:
        f.write(f"{'='*60}\n")
        f.write(f"SINGLE IMAGE TEST RESULTS - {result_file}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint  : {args.checkpoint}\n")
        f.write(f"Image path  : {args.image_path}\n")
        f.write(f"Color mode  : {args.color_mode}\n")
        f.write(f"Image size  : {img_size[0]}x{img_size[1]}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Predicted Class : {class_names[predicted_class]}\n")
        f.write(f"Confidence      : {confidence:.2f}%\n")
        f.write(f"\nProbability Distribution:\n")
        f.write(f"  Cover : {probs[0]*100:.2f}%\n")
        f.write(f"  Stego : {probs[1]*100:.2f}%\n")

if __name__ == "__main__":
    main()
