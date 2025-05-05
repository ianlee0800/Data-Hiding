# 預測誤差嵌入 (PEE) 系統

## 概述

預測誤差嵌入 (PEE) 系統是一種先進的數位影像資料隱藏技術實現。它通過預測誤差操作來嵌入秘密信息，同時保持影像的視覺品質。系統整合了多種嵌入策略、預測方法和全面的視覺化工具。

## 功能特色

- **多種嵌入方法**：
  - **旋轉型**：在不同角度旋轉影像區塊後嵌入資料
  - **分割型**：使用影像分割技術（區塊型或交錯型）
  - **四叉樹型**：根據內容複雜度自適應分割影像

- **多種預測方法**：
  - **PROPOSED**：自定義加權預測演算法
  - **MED**：中值邊緣檢測預測器
  - **GAP**：梯度調整預測
  - **RHOMBUS**：基於鄰近像素的模式預測

- **影像支援**：
  - 自動灰階/彩色影像檢測
  - 針對灰階和彩色影像的專門處理
  - 針對彩色影像的通道特定處理

- **分析與視覺化**：
  - 效能指標（PSNR、SSIM、直方圖相關性）
  - 嵌入過程分析的全面視覺化
  - 四叉樹方法的區塊大小分佈圖
  - 差異直方圖（嵌入前、移位後、嵌入後）
  - 四叉樹方法的區塊特定視覺化

- **評估工具**：
  - 精確測量模式，用於詳細的容量-失真分析
  - 方法比較功能
  - 多預測器評估

## 相依性

系統需要以下 Python 函式庫：

```
numpy
cupy（用於 GPU 加速）
cv2（OpenCV）
matplotlib
tqdm
pandas
prettytable
numba
```

## 使用方法

### 基本使用

主要參數可直接在程式碼中配置：

```python
# 基本參數
imgName = "Male"           # 影像名稱
filetype = "tiff"         # 影像檔案類型
total_embeddings = 5      # 總嵌入階段數

# 方法選擇
method = "quadtree"       # 選項："rotation"、"split"、"quadtree"
prediction_method_str = "PROPOSED"  # 選項："PROPOSED"、"MED"、"GAP"、"RHOMBUS"、"ALL"

# 預測器設定
predictor_ratios = {
    "PROPOSED": 0.5,      # PROPOSED 預測器的 1 比例
    "MED": 1.0,           # MED 預測器的 1 比例
    "GAP": 0.7,           # GAP 預測器的 1 比例
    "RHOMBUS": 0.9        # RHOMBUS 預測器的 1 比例
}

# 方法特定參數
split_size = 64           # 用於旋轉和分割方法
block_base = False        # 用於分割方法
quad_tree_params = {
    'min_block_size': 16,  # 四叉樹的最小區塊大小
    'variance_threshold': 300  # 四叉樹分割閾值
}
```

執行程式：

```
python main.py
```

### 進階使用

#### 精確測量模式

用於詳細的容量-失真分析：

```python
use_precise_measurement = True
stats_segments = 20       # 測量點數量
# 或
step_size = 100000        # 測量步長（位元）
```

#### 方法比較

比較不同的嵌入方法：

```python
use_method_comparison = True
methods_to_compare = ["rotation", "quadtree"]
comparison_predictor = "proposed"
```

## 方法說明

### 嵌入方法

1. **旋轉方法**
   - 將影像分割成大小為 `split_size × split_size` 的子影像
   - 對每個階段應用旋轉並使用預測誤差嵌入資料
   - 隨著每個階段的旋轉角度增加，提供額外容量

2. **分割方法**
   - 使用區塊型或交錯型分割影像
   - 可對單個區塊使用隨機旋轉以增強安全性
   - 支援靈活的容量控制

3. **四叉樹方法**
   - 根據內容複雜度自適應分割影像
   - 使用不同大小的區塊（16×16 到 512×512）進行最佳嵌入
   - 允許內容自適應資料隱藏，提升視覺品質

### 預測方法

1. **PROPOSED**
   - 使用可優化權重的加權預測方案
   - 支援基於局部內容的自適應嵌入層級

2. **MED（中值邊緣檢測）**
   - 基於邊緣檢測原理預測像素
   - 適用於有方向性紋理的區域

3. **GAP（梯度調整預測）**
   - 使用梯度信息調整預測
   - 在紋理豐富的區域更準確

4. **RHOMBUS**
   - 基於鄰近像素的模式預測
   - 適用於平滑且具有漸變的區域

## 輸出與視覺化

系統生成的大量輸出檔案按以下結構組織：

```
./Prediction_Error_Embedding/outcome/
├── data/
│   └── [image_name]/
│       └── [method]_[predictor]_final_results.npy
├── histogram/
│   └── [image_name]/
│       └── [method]/
│           ├── original_histogram.png
│           ├── stage_[i]_histogram.png
│           ├── difference_histograms/
│           │   ├── [method]_stage[i]_error_before.png
│           │   ├── [method]_stage[i]_error_shifted.png
│           │   └── [method]_stage[i]_error_after.png
│           └── block_histograms/ (用於四叉樹)
├── image/
│   └── [image_name]/
│       └── [method]/
│           ├── original.png
│           ├── stage_[i]_result.png
│           ├── final_result.png
│           ├── original_vs_final.png
│           ├── final_heatmap.png
│           ├── block_size_visualizations/ (用於四叉樹)
│           │   └── stage_[i]_blocks_[size]x[size].png
│           └── [方法特定目錄]
└── plots/
    └── [image_name]/
        └── [method]/
            ├── bpp_psnr_curve.png
            ├── payload_distribution.png
            ├── metrics_comparison.png
            └── [方法特定圖表]
```

### 關鍵視覺化

1. **差異直方圖**
   - 嵌入前：初始預測誤差分佈
   - 移位後：移位誤差分佈（模擬）
   - 嵌入後：資料隱藏後的最終誤差分佈
   - 比較：三種直方圖重疊，便於比較

2. **區塊大小視覺化（四叉樹）**
   - 針對每個區塊大小（16×16、32×32等）的獨立視覺化
   - 僅在特定大小的區塊中顯示原始內容
   - 其他區域顯示為黑色背景
   - 包含區塊數量和視覺網格

3. **效能指標**
   - BPP-PSNR 曲線用於容量-失真分析
   - 階段性指標比較
   - 彩色影像的通道特定指標

## 程式結構

專案由以下 Python 模組組成：

- **main.py**：主執行檔案，包含參數設定和流程控制
- **color.py**：彩色影像處理和分析函數
- **common.py**：通用實用函數和指標計算
- **embedding.py**：不同方法的核心嵌入函數
- **image_processing.py**：影像操作和處理實用工具
- **pee.py**：預測誤差嵌入演算法實現
- **quadtree.py**：基於四叉樹的自適應分割
- **utils.py**：額外的輔助函數和分析工具
- **visualization.py**：結果分析的視覺化函數

## 應用範例

系統可用於各種應用：

1. **資料隱藏**：在數位影像中嵌入機密資訊
2. **隱寫術**：建立含有隱藏訊息的影像
3. **影像處理研究**：預測誤差分佈分析
4. **效能評估**：比較不同的嵌入和預測方法
5. **教育目的**：了解影像處理和資料隱藏技術

## 注意事項

- 透過 CUDA 支援 GPU 加速以加快處理速度
- 針對大型影像優化記憶體管理
- 詳細記錄提供嵌入過程的全面資訊
- 近似和精確測量模式允許靈活評估
- 所有視覺化自動儲存以供後續分析

## 延伸閱讀

有關預測誤差嵌入技術的更多資訊，請參考：
- 預測誤差擴展的學術論文
- 影像隱寫術和資料隱藏資源
- CuPy、Numba 和 OpenCV 函式庫的文件