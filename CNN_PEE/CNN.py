import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from scipy.signal import convolve2d
import time
import torch.cuda.amp as amp

# Set environment variable to ignore multiprocessing-related warnings
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        transform = transforms.RandomResizedCrop(self.size, scale=self.scale, ratio=self.ratio)
        return transform(img)

def preprocess_image(image):
    h, w = image.shape
    parts = [
        image[0:h:2, 0:w:2],
        image[0:h:2, 1:w:2],
        image[1:h:2, 0:w:2],
        image[1:h:2, 1:w:2]
    ]
    
    filter_kernel = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 6, 4, 2],
        [3, 6, 9, 6, 3],
        [2, 4, 6, 4, 2],
        [1, 2, 3, 2, 1]
    ]) / 81  # Pre-compute normalization factor

    preprocessed_image = np.zeros_like(image)
    for i, part in enumerate(parts):
        filtered_part = convolve2d(part, filter_kernel, mode='same', boundary='symmetric')
        preprocessed_image[i//2::2, i%2::2] = filtered_part

    return preprocessed_image

def preprocess_image_tensor(x):
    return torch.from_numpy(preprocess_image(x.squeeze().numpy())).unsqueeze(0)

class ILSVRCSubset(Dataset):
    def __init__(self, root_dir, num_images=3000, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths(num_images)

    def _get_image_paths(self, num_images):
        train_dir = os.path.join(self.root_dir, 'ILSVRC', 'Data', 'DET', 'train')
        val_dir = os.path.join(self.root_dir, 'ILSVRC', 'Data', 'DET', 'val')
        
        all_image_paths = []
        for directory in [train_dir, val_dir]:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        all_image_paths.append(os.path.join(root, file))
        
        return random.sample(all_image_paths, min(num_images, len(all_image_paths)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def load_data(root_dir, num_images=3000, batch_size=8):
    print("Starting data loading...")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Keep original image size
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Lambda(preprocess_image_tensor),
    ])

    dataset = ILSVRCSubset(root_dir, num_images=num_images, transform=transform)
    print(f"Dataset size: {len(dataset)} images")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    print("Data loading completed")
    return train_loader, val_loader

class FeatureBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FeatureBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        return x

class ImprovedAdaptiveCNNPredictor(nn.Module):
    def __init__(self):
        super(ImprovedAdaptiveCNNPredictor, self).__init__()
        self.feature_blocks = nn.ModuleList([
            FeatureBlock(1, 32, 3),
            FeatureBlock(1, 32, 5),
            FeatureBlock(1, 32, 7)
        ])
        
        self.conv1x1 = nn.Conv2d(96, 32, kernel_size=1)
        
        self.prediction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.1)
            ) for _ in range(15)
        ])
        
        self.reconstruction = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        features = torch.cat([block(x) for block in self.feature_blocks], dim=1)
        features = self.conv1x1(features)
        
        for layer in self.prediction_layers:
            features = layer(features) + features  # residual connection
        
        return self.reconstruction(features)

class CustomLoss(nn.Module):
    def __init__(self, lambda_reg=1e-3):
        super(CustomLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, output, target, model):
        mse_loss = torch.mean((output - target) ** 2)
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        return mse_loss + self.lambda_reg * l2_reg

def train_model(model, train_loader, val_loader, num_epochs, device, accumulation_steps=4):
    print("Starting model training...")
    criterion = CustomLoss(lambda_reg=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = amp.GradScaler()

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\nStarting epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, inputs in enumerate(train_loader):
            batch_start_time = time.time()
            inputs = inputs.to(device, non_blocking=True)
            
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, inputs, model) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            if batch_idx % 10 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Batch Time: {time.time() - batch_start_time:.2f}s')

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        print("Starting validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                with amp.autocast():
                    outputs = model(inputs)
                    val_loss += criterion(outputs, inputs, model).item()
        val_loss /= len(val_loader)

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]}, Time: {epoch_time:.2f}s')

        if (epoch + 1) % 5 == 0:
            model_path = f'./CNN_PEE/model/model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed, total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    print("Program execution started...")
    torch.multiprocessing.set_start_method('spawn')

    print("Checking CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = ImprovedAdaptiveCNNPredictor().to(device)
    
    print("Setting data path...")
    ilsvrc_root = r"C:\Users\ianle\Downloads\ILSVRC2017_DET"
    
    print("Loading data...")
    train_loader, val_loader = load_data(ilsvrc_root, num_images=3000, batch_size=8)

    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    print("Starting training process...")
    train_model(model, train_loader, val_loader, num_epochs=50, device=device, accumulation_steps=4)

    print("Program execution completed")