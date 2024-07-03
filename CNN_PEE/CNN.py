import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os
import numpy as np
from scipy.signal import convolve2d

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
    ])
    filter_kernel = filter_kernel / np.sum(filter_kernel)
    
    preprocessed_parts = []
    for part in parts:
        filtered_part = convolve2d(part, filter_kernel, mode='same', boundary='symmetric')
        preprocessed_parts.append(filtered_part)
    
    preprocessed_image = np.zeros_like(image)
    preprocessed_image[0:h:2, 0:w:2] = preprocessed_parts[0]
    preprocessed_image[0:h:2, 1:w:2] = preprocessed_parts[1]
    preprocessed_image[1:h:2, 0:w:2] = preprocessed_parts[2]
    preprocessed_image[1:h:2, 1:w:2] = preprocessed_parts[3]
    
    return preprocessed_image

def load_data():
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 直接調整到目標大小
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = CIFAR10(root='./CNN_PEE/data', train=True, download=True, transform=transform)
    
    train_size = 3000
    val_size = 1000
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, len(dataset)-train_size-val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader

class AdaptiveCNNPredictor(nn.Module):
    def __init__(self):
        super(AdaptiveCNNPredictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        features = self.features(x)
        return self.predictor(features)

class CustomLoss(nn.Module):
    def __init__(self, lambda_reg=1e-3):
        super(CustomLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, output, target, model):
        mse_loss = torch.mean((output - target) ** 2)
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        return mse_loss + self.lambda_reg * l2_reg

def calculate_metrics(model, dataloader, device):
    model.eval()
    total_mean_error = 0
    total_var_error = 0
    n_samples = 0
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            error = (outputs - inputs).abs()
            
            total_mean_error += error.mean().item() * inputs.size(0)
            total_var_error += error.var().item() * inputs.size(0)
            n_samples += inputs.size(0)
    
    a_mean = total_mean_error / n_samples
    a_var = total_var_error / n_samples
    
    return a_mean, a_var

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = CustomLoss(lambda_reg=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # 驗證
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, inputs, model).item()
        val_loss /= len(val_loader)

        # 計算評估指標
        a_mean, a_var = calculate_metrics(model, val_loader, device)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'A-Mean: {a_mean:.4f}, '
              f'A-Var: {a_var:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]}')

def predict(model, image, device):
    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        prediction = model(image)
    return prediction.squeeze().cpu().numpy()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveCNNPredictor().to(device)
    train_loader, val_loader = load_data()

    # 訓練模型
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    # 創建儲存模型的目錄
    os.makedirs("./CNN_PEE/model", exist_ok=True)

    # 保存模型
    torch.save(model.state_dict(), './CNN_PEE/model/adaptive_cnn_predictor.pth')
    print("模型已保存至 './CNN_PEE/model/adaptive_cnn_predictor.pth'")