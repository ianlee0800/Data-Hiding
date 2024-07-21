import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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

def load_model(model_path):
    """加載預訓練的模型"""
    model = ImprovedAdaptiveCNNPredictor()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 設置為評估模式
    return model

def preprocess_image(image):
    """預處理圖像以輸入到模型"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)

def postprocess_image(tensor):
    """將模型輸出轉換回numpy數組"""
    output = tensor.squeeze().cpu().detach().numpy()
    output = (output * 0.5 + 0.5) * 255
    return np.clip(output, 0, 255).astype(np.uint8)

def generate_predict_image(img, model):
    """使用CNN模型生成預測圖像"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    with torch.no_grad():
        input_tensor = preprocess_image(img).to(device)
        output_tensor = model(input_tensor)
    
    return postprocess_image(output_tensor)

# 如果需要直接運行此文件進行測試
if __name__ == "__main__":
    # 這裡可以添加一些測試代碼
    # 例如:
    model_path = "./CNN_PEE/model/adaptive_cnn_predictor.pth"
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        
        # 創建一個測試圖像
        test_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # 生成預測圖像
        predicted_image = generate_predict_image(test_image, model)
        
        print(f"Original image shape: {test_image.shape}")
        print(f"Predicted image shape: {predicted_image.shape}")
        print("Prediction completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")