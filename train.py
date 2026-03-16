"""Training script for XuNet with proper checkpoint management."""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from dataset.dataset import DatasetLoad
from model.model import XuNet
from opts.options import arguments

# Parse arguments
opt = arguments()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Determine input channels based on color mode
in_channels = 3 if opt.color_mode == "RGB" else 1

# Initialize model
model = XuNet(in_channels=in_channels).to(device)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=opt.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=5e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.2,
    patience=5,
    threshold=0.0001,
    threshold_mode='rel',
    cooldown=0,
    min_lr=1e-6,
    eps=1e-08
)

# Data transformations
transform = transforms.Compose([transforms.ToTensor()])

# Load datasets
train_dataset = DatasetLoad(
    opt.cover_path,
    opt.stego_path,
    opt.train_size,
    transform=transform,
    file_extension=opt.file_extension,
    color_mode=opt.color_mode
)

valid_dataset = DatasetLoad(
    opt.valid_cover_path,
    opt.valid_stego_path,
    opt.val_size,
    transform=transform,
    file_extension=opt.file_extension,
    color_mode=opt.color_mode
)

train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

# ============================================================
# LOGGING SYSTEM
# ============================================================

class Logger:
    """Dual logger to write to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(checkpoint_dir):
    """Setup logging to file in logs/ directory."""
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    resolution = os.path.basename(checkpoint_dir.rstrip('/\\'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"training_{resolution}_{timestamp}.log")
    
    sys.stdout = Logger(log_file)
    
    print("="*60)
    print(f"TRAINING LOG - {resolution.upper()}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print("="*60)
    print()
    
    return log_file

# ============================================================
# CHECKPOINT MANAGEMENT (FIXED)
# ============================================================

def create_checkpoint_dir(checkpoint_dir):
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"✅ Checkpoint directory: {checkpoint_dir}")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # List all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('net_') and f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    return latest_checkpoint

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load checkpoint and return start epoch."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No checkpoints found! Starting training from scratch...")
        return 0
    
    print(f"✅ Found checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    print(f"✅ Resuming training from epoch {start_epoch + 1}")
    print(f"   Previous train loss: {checkpoint.get('train_loss', 'N/A'):.5f}")
    print(f"   Previous valid loss: {checkpoint.get('valid_loss', 'N/A'):.5f}")
    print(f"   Previous train acc: {checkpoint.get('train_acc', 'N/A'):.2f}%")
    print(f"   Previous valid acc: {checkpoint.get('valid_acc', 'N/A'):.2f}%")
    
    return start_epoch

def save_checkpoint(checkpoint_dir, epoch, model, optimizer, train_loss, valid_loss, train_acc, valid_acc):
    """Save checkpoint to the correct directory."""
    checkpoint_path = os.path.join(checkpoint_dir, f"net_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'train_acc': train_acc,
        'valid_acc': valid_acc
    }, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")

# ============================================================
# TRAINING SETUP
# ============================================================

# Setup logging system
log_file = setup_logging(opt.checkpoints_dir)

# Create checkpoint directory
create_checkpoint_dir(opt.checkpoints_dir)

# Find and load latest checkpoint
latest_checkpoint = find_latest_checkpoint(opt.checkpoints_dir)
start_epoch = load_checkpoint(latest_checkpoint, model, optimizer)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Resolution: Based on dataset")
print(f"Color mode: {opt.color_mode}")
print(f"Input channels: {in_channels}")
print(f"Train size: {opt.train_size} pairs")
print(f"Validation size: {opt.val_size} pairs")
print(f"Batch size: {opt.batch_size}")
print(f"Learning rate: {opt.lr}")
print(f"Total epochs: {opt.num_epochs}")
print(f"Starting from epoch: {start_epoch + 1}")
print(f"Checkpoint directory: {opt.checkpoints_dir}")
print("="*60 + "\n")

# ============================================================
# TRAINING LOOP
# ============================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, data in enumerate(pbar):
        # Get cover and stego images
        covers = data['cover'].to(device)
        stegos = data['stego'].to(device)
        
        # Concatenate cover and stego into single batch
        images = torch.cat([covers, stegos], dim=0)
        
        # Create labels: 0 for cover, 1 for stego
        batch_size = covers.size(0)
        labels = torch.cat([
            torch.zeros(batch_size, dtype=torch.long, device=device),
            torch.ones(batch_size, dtype=torch.long, device=device)
        ], dim=0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, valid_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in tqdm(valid_loader, desc="Validation"):
            # Get cover and stego images
            covers = data['cover'].to(device)
            stegos = data['stego'].to(device)
            
            # Concatenate cover and stego into single batch
            images = torch.cat([covers, stegos], dim=0)
            
            # Create labels: 0 for cover, 1 for stego
            batch_size = covers.size(0)
            labels = torch.cat([
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.ones(batch_size, dtype=torch.long, device=device)
            ], dim=0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(valid_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Main training loop
for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{opt.num_epochs}")
    print(f"{'='*60}")
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, opt.num_epochs)
    
    # Validation
    valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, device)
    
    # Learning rate scheduling
    scheduler.step(valid_loss)
    
    # Print epoch summary
    print(f"\n📊 Epoch {epoch} Summary:")
    print(f"   Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%")
    print(f"   Valid Loss: {valid_loss:.5f} | Valid Acc: {valid_acc:.2f}%")
    print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Save checkpoint
    save_checkpoint(
        opt.checkpoints_dir,
        epoch,
        model,
        optimizer,
        train_loss,
        valid_loss,
        train_acc,
        valid_acc
    )

print("\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"All checkpoints saved in: {opt.checkpoints_dir}")
print(f"Training log saved in: {log_file}")
print(f"Best model: Check validation accuracy in logs")
print("="*60)

# Restore stdout
if hasattr(sys.stdout, 'log'):
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
