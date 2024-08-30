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
from torch.utils.tensorboard import SummaryWriter

# Set environment variable to ignore multiprocessing-related warnings
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

# Set device type and device
DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(DEVICE_TYPE)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle)

class LargeRandomRotation(object):
    def __init__(self, degrees=90):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return transforms.functional.rotate(img, angle, expand=True)

class ILSVRCSubset(Dataset):
    def __init__(self, root_dir, num_images=10, transform=None):
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

def load_data(root_dir, num_images=3000, batch_size=32):
    print("Starting data loading with large angle rotation...")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        LargeRandomRotation(degrees=90),  # 大角度隨機旋轉
        transforms.CenterCrop((512, 512)),  # 裁剪以去除旋轉導致的黑邊
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
        
        self.prediction_layers = nn.Sequential(*[
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

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9
        }
    return {'allocated': 0, 'cached': 0}

def train_model(model, train_loader, val_loader, num_epochs, device, accumulation_steps=8):
    print("Starting model training...")
    criterion = CustomLoss(lambda_reg=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda') if DEVICE_TYPE == 'cuda' else None

    start_time = time.time()

    for epoch in range(num_epochs):
        if DEVICE_TYPE == 'cuda':
            torch.cuda.empty_cache()  # 清理 GPU 缓存
        model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            
            if DEVICE_TYPE == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs, model) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, inputs, model) / accumulation_steps
                loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                if DEVICE_TYPE == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            batch_count += 1

            if batch_idx % 50 == 0:
                print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
                if DEVICE_TYPE == 'cuda':
                    print(f'  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB')

        avg_train_loss = total_loss / batch_count

        print("Starting validation...")
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                if DEVICE_TYPE == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        val_loss = criterion(outputs, inputs, model)
                else:
                    outputs = model(inputs)
                    val_loss = criterion(outputs, inputs, model)
                total_val_loss += val_loss.item()
                val_batch_count += 1
        avg_val_loss = total_val_loss / val_batch_count

        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        if DEVICE_TYPE == 'cuda':
            print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB')

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model_path = f'./CNN_PEE/model/model_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, model_path, _use_new_zipfile_serialization=False)
            print(f"Model saved to {model_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed, total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    print("Program execution started...")
    torch.multiprocessing.set_start_method('spawn')

    print("Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = ImprovedAdaptiveCNNPredictor().to(device)
    
    print("Setting data path...")
    ilsvrc_root = r"C:\Users\Ian Lee\Downloads\ILSVRC2017_DET"
    
    print("Loading data...")
    train_loader, val_loader = load_data(ilsvrc_root, num_images=3000, batch_size=16)  # 减小批次大小

    if DEVICE_TYPE == 'cuda':
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")

    print("Starting training process...")
    try:
        train_model(model, train_loader, val_loader, num_epochs=50, device=device, accumulation_steps=8)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Try reducing batch size or model size.")
        else:
            print(f"An error occurred: {e}")

    if DEVICE_TYPE == 'cuda':
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")

    print("Program execution completed")