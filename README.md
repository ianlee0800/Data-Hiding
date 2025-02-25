# 數據隱藏系統使用說明

## 新功能說明

本次更新新增了以下功能：

1. **新增菱形預測器 (Rhombus Predictor)** - 使用上、下、左、右四個相鄰像素進行加權預測，增加了一種新的預測選擇。

2. **最大嵌入統計** - 支持使用圖像的最大嵌入容量，並將其均勻分成15段進行統計，方便論文中繪製折線圖。

3. **多預測方法自動運行與比較** - 可以一次性運行所有預測方法（PROPOSED、MED、GAP、RHOMBUS），並自動生成比較圖表。

4. **CSV數據輸出** - 所有統計數據均以CSV格式保存，方便在論文撰寫過程中使用。

## 使用方法

### 基本使用

1. 打開 `main.py` 文件
2. 在文件開頭的參數設置部分調整需要的參數
3. 使用 VS Code 的 "Run Code" 按鈕執行代碼

### 關鍵參數

在 `main.py` 中，您可以調整以下關鍵參數：

```python
# 基本參數設置
imgName = "male"            # 圖像名稱
filetype = "png"            # 圖像檔案類型
total_embeddings = 5        # 總嵌入次數
ratio_of_ones = 0.5         # 嵌入數據中1的比例

# 預測方法選擇
# 可選：PROPOSED, MED, GAP, RHOMBUS, ALL (ALL表示運行所有方法並生成比較)
prediction_method_str = "ALL"

# 方法選擇
method = "quadtree"         # 可選："rotation", "split", "quadtree"
```

### 運行所有預測方法並生成比較

設置 `prediction_method_str = "ALL"` 即可自動運行所有四種預測方法並生成比較圖表。

### 調整統計分段數量

默認將總嵌入容量分成15段進行統計，您可以通過調整以下參數來更改：

```python
# 統計分段數量
stats_segments = 15
```

## 輸出說明

運行完成後，程序會在以下目錄生成各種輸出文件：

### 標準輸出

- `./pred_and_QR/outcome/image/{imgName}/` - 處理後的圖像
- `./pred_and_QR/outcome/histogram/{imgName}/` - 直方圖
- `./pred_and_QR/outcome/plots/{imgName}/` - 折線圖和統計數據

### 多預測方法比較輸出（當選擇 ALL 時）

- `./pred_and_QR/outcome/plots/{imgName}/comparison/` - 比較結果目錄
  - `unified_bpp_psnr.png` - 包含所有預測方法的BPP-PSNR曲線
  - `unified_bpp_ssim.png` - 包含所有預測方法的BPP-SSIM曲線
  - `combined_statistics.csv` - 所有預測方法的合併統計數據
  - `wide_format_psnr.csv` - 適合直接用於論文表格的寬格式PSNR數據
  - `wide_format_ssim.csv` - 適合直接用於論文表格的寬格式SSIM數據
  - `summary_results.csv` - 各預測方法的最終結果摘要

## 菱形預測器說明

新增的菱形預測器 (Rhombus Predictor) 使用上、下、左、右四個點的加權平均來預測當前像素值。其工作原理如下：

1. 使用上下左右四個相鄰像素進行預測
2. 自動檢測水平和垂直邊緣，並根據邊緣特性調整預測行為
3. 當檢測到水平邊緣時，優先使用水平方向的像素
4. 當檢測到垂直邊緣時，優先使用垂直方向的像素

這種預測模式在某些類型的圖像上可能表現更好，尤其是具有明顯水平或垂直邊緣的圖像。