import numpy as np
import struct
import cupy as cp
import matplotlib.pyplot as plt
from common import *
from pee import *
from prettytable import PrettyTable

def generate_embedding_data(total_embeddings, sub_images_per_stage, max_capacity_per_subimage, 
                           ratio_of_ones=0.5, target_payload_size=-1):
    """
    更靈活的嵌入數據生成函數，不強制平均分配 payload
    
    Parameters:
    -----------
    total_embeddings : int
        總嵌入階段數
    sub_images_per_stage : int
        每個stage的子圖像數量
    max_capacity_per_subimage : int
        每個子圖像的最大容量
    ratio_of_ones : float, optional
        生成數據中1的比例，默認為0.5
    target_payload_size : int, optional
        目標總payload大小，設為-1或0時使用最大容量
        
    Returns:
    --------
    dict
        包含每個stage的數據生成資訊
    """
    # 如果沒有指定目標payload，使用最大容量模式
    if target_payload_size <= 0:
        max_stage_payload = sub_images_per_stage * max_capacity_per_subimage
        stage_data = []
        for _ in range(total_embeddings):
            sub_data_list = []
            for _ in range(sub_images_per_stage):
                sub_data = generate_random_binary_array(max_capacity_per_subimage, ratio_of_ones)
                sub_data_list.append(sub_data)
            stage_data.append({
                'sub_data': sub_data_list,
                'remaining_target': 0
            })
        return {
            'stage_data': stage_data,
            'total_target': max_stage_payload * total_embeddings
        }
    
    # 使用指定的payload size
    # 為每個stage生成足夠大的數據，讓它們能夠靈活地達到目標payload
    total_remaining = target_payload_size
    stage_data = []
    
    # 為每個stage分配潛在的最大容量
    potential_capacity_per_stage = max_capacity_per_subimage * sub_images_per_stage
    
    for stage in range(total_embeddings):
        sub_data_list = []
        
        # 為每個子圖像生成數據
        for sub_img in range(sub_images_per_stage):
            # 計算這個子圖像可能需要的最大數據量
            max_possible = min(max_capacity_per_subimage, total_remaining)
            
            # 如果是最後一個stage的最後一個子圖像，確保生成足夠的數據
            if stage == total_embeddings - 1 and sub_img == sub_images_per_stage - 1:
                sub_data = generate_random_binary_array(total_remaining, ratio_of_ones)
            else:
                sub_data = generate_random_binary_array(max_possible, ratio_of_ones)
            
            sub_data_list.append(sub_data)
        
        stage_data.append({
            'sub_data': sub_data_list,
            'remaining_target': total_remaining  # 記錄當前階段還需要嵌入多少數據
        })
    
    return {
        'stage_data': stage_data,
        'total_target': target_payload_size
    }

# 保留原有函數以保持向後兼容性
def generate_random_binary_array(size, ratio_of_ones=0.5):
    """生成指定大小的随机二进制数组，可调整1的比例"""
    return np.random.choice([0, 1], size=size, p=[1-ratio_of_ones, ratio_of_ones])

def encode_pee_info(pee_info):
    encoded = struct.pack('B', pee_info['total_rotations'])
    for stage in pee_info['stages']:
        for block in stage['block_params']:
            # 编码权重（每个权重使用4位，4个权重共16位）
            weights_packed = sum(w << (4 * i) for i, w in enumerate(block['weights']))
            encoded += struct.pack('>HBH', 
                weights_packed,
                block['EL'],
                block['payload']
            )
    return encoded

def decode_pee_info(encoded_data):
    total_rotations = struct.unpack('B', encoded_data[:1])[0]
    stages = []
    offset = 1

    for _ in range(total_rotations):
        block_params = []
        for _ in range(4):  # 每次旋转有4个块
            weights_packed, EL, payload = struct.unpack('>HBH', encoded_data[offset:offset+5])
            weights = [(weights_packed >> (4 * i)) & 0xF for i in range(4)]
            block_params.append({
                'weights': weights,
                'EL': EL,
                'payload': payload
            })
            offset += 5
        stages.append({'block_params': block_params})

    return {
        'total_rotations': total_rotations,
        'stages': stages
    }

def create_pee_info_table(pee_stages, use_different_weights, total_pixels, split_size, quad_tree=False):
    """
    創建PEE資訊表格
    
    Parameters:
    -----------
    pee_stages : list
        包含所有PEE階段資訊的列表
    use_different_weights : bool
        是否對每個子圖像使用不同的權重
    total_pixels : int
        圖像總像素數
    split_size : int
        分割大小
    quad_tree : bool, optional
        是否使用quad tree模式
    
    Returns:
    --------
    PrettyTable
        格式化的表格
    """
    table = PrettyTable()
    
    if quad_tree:
        # Quad tree模式的表格欄位
        table.field_names = [
            "Embedding", "Block Size", "Block Position", "Payload", "BPP",
            "PSNR", "SSIM", "Hist Corr", "Weights", "EL", "Note"
        ]
    else:
        # 原有模式的表格欄位
        table.field_names = [
            "Embedding", "Sub-image", "Payload", "BPP", "PSNR", "SSIM",
            "Hist Corr", "Weights", "EL", "Rotation", "Note"
        ]
    
    for stage in pee_stages:
        # 添加整體stage資訊
        if quad_tree:
            # Quad tree模式的整體資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-",
                "-",
                stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-",
                "-",
                "Stage Summary"
            ])
        else:
            # 原有模式的整體資訊
            table.add_row([
                f"{stage['embedding']} (Overall)",
                "-",
                stage['payload'],
                f"{stage['bpp']:.4f}",
                f"{stage['psnr']:.2f}",
                f"{stage['ssim']:.4f}",
                f"{stage['hist_corr']:.4f}",
                "-",
                "-",
                "-",
                "Stage Summary"
            ])
        
        table.add_row(["-" * 5] * len(table.field_names))
        
        if quad_tree:
            # 處理quad tree模式的區塊資訊
            for size in sorted(stage['block_info'].keys(), key=int, reverse=True):
                blocks = stage['block_info'][size]['blocks']
                for block in blocks:
                    block_pixels = block['size'] * block['size']
                    table.add_row([
                        stage['embedding'],
                        f"{block['size']}x{block['size']}",
                        f"({block['position'][0]}, {block['position'][1]})",
                        block['payload'],
                        f"{block['payload'] / block_pixels:.4f}",
                        f"{block['psnr']:.2f}",
                        f"{block['ssim']:.4f}",
                        f"{block['hist_corr']:.4f}",
                        ", ".join([f"{w:.2f}" for w in block['weights']]),
                        block.get('EL', '-'),
                        "Different weights" if use_different_weights else ""
                    ])
        else:
            # 處理原有模式的區塊資訊
            total_blocks = split_size * split_size
            sub_image_pixels = total_pixels // total_blocks
            
            for i, block in enumerate(stage['block_params']):
                table.add_row([
                    stage['embedding'] if i == 0 else "",
                    i,
                    block['payload'],
                    f"{block['payload'] / sub_image_pixels:.4f}",
                    f"{block['psnr']:.2f}",
                    f"{block['ssim']:.4f}",
                    f"{block['hist_corr']:.4f}",
                    ", ".join([f"{w:.2f}" for w in block['weights']]),
                    block['EL'],
                    f"{block['rotation']}°",
                    "Different weights" if use_different_weights else ""
                ])
        
        table.add_row(["-" * 5] * len(table.field_names))
    
    return table

def analyze_and_plot_results(bpp_psnr_data, imgName, split_size):
    """
    分析結果並繪製圖表
    """
    plt.figure(figsize=(12, 8))
    
    # 繪製 BPP-PSNR 曲線
    bpps = [data['bpp'] for data in bpp_psnr_data]
    psnrs = [data['psnr'] for data in bpp_psnr_data]
    
    plt.plot(bpps, psnrs, 'b.-', linewidth=2, markersize=8, 
             label=f'Split Size: {split_size}x{split_size}')
    
    plt.xlabel('Bits Per Pixel (BPP)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'BPP-PSNR Curve for {imgName}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 添加數據標籤
    for i, (bpp, psnr) in enumerate(zip(bpps, psnrs)):
        plt.annotate(f'Stage {i}\n({bpp:.3f}, {psnr:.2f})',
                    (bpp, psnr), textcoords="offset points",
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    return plt.gcf()

# 在 utils.py 中添加
def calculate_metrics_with_rotation(original_img, stage_img, current_rotation):
    """
    計算考慮旋轉的圖像品質指標
    
    Parameters:
    -----------
    original_img : numpy.ndarray
        原始圖像
    stage_img : numpy.ndarray
        處理後的圖像
    current_rotation : int
        當前旋轉角度（度數）
    
    Returns:
    --------
    tuple
        (psnr, ssim, hist_corr) 三個品質指標
    """
    # 確保輸入是numpy array
    if isinstance(original_img, cp.ndarray):
        original_img = cp.asnumpy(original_img)
    if isinstance(stage_img, cp.ndarray):
        stage_img = cp.asnumpy(stage_img)
    
    # 如果需要旋轉，將圖像旋轉回原始方向
    if current_rotation != 0:
        k = (-current_rotation // 90) % 4
        stage_img = np.rot90(stage_img, k=k)
    
    # 計算各種指標
    psnr = calculate_psnr(original_img, stage_img)
    ssim = calculate_ssim(original_img, stage_img)
    hist_corr = histogram_correlation(
        np.histogram(original_img, bins=256, range=(0, 255))[0],
        np.histogram(stage_img, bins=256, range=(0, 255))[0]
    )
    
    return psnr, ssim, hist_corr

