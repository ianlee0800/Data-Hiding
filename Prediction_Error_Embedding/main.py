
"""
重構後的 PEE 主程式
整合所有副程式功能，大幅簡化代碼結構
"""

import json
import os
import warnings
from numba.core.errors import NumbaPerformanceWarning

# 核心功能導入
from image_processing import (
    PEEVisualizationSystem, 
    PredictionMethod
)
import cv2
from utils import (
    get_output_directories, 
    create_all_directories,
    cleanup_memory,
    find_image_path
)
from measurement import (
    run_precise_measurements,
    run_simplified_precise_measurements, 
    run_multi_predictor_precise_measurements,
    run_method_comparison
)
from embedding import (
    pee_process_with_rotation_cuda,
    pee_process_with_split_cuda
)
from quadtree import pee_process_with_quadtree_cuda

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# =============================================================================
# 配置管理系統
# =============================================================================

class ConfigManager:
    """統一的配置管理器"""
    
    DEFAULT_CONFIG = {
        "experiment": {
            "name": "PEE_Experiment",
            "description": "Prediction Error Embedding Experiment"
        },
        "image": {
            "name": "F16",
            "filetype": "tiff"
        },
        "embedding": {
            "total_embeddings": 4,
            "el_mode": 0,
            "use_different_weights": True,
            "predictor_ratios": {
                "PROPOSED": 0.5,
                "MED": 0.5,
                "GAP": 0.5,
                "RHOMBUS": 0.5
            }
        },
        "method": {
            "name": "quadtree",
            "rotation": {"split_size": 2},
            "split": {"split_size": 2, "block_base": False},
            "quadtree": {
                "min_block_size": 16,
                "variance_threshold": 300,
                "adaptive_threshold": True,
                "search_mode": "balanced",
                "target_bpp_for_search": 0.8,
                "target_psnr_for_search": 35.0
            }
        },
        "measurement": {
            "use_precise_measurement": True,
            "use_method_comparison": False,
            "stats_segments": 20,
            "step_size": 100000,
            "methods_to_compare": ["rotation", "quadtree"],
            "comparison_predictor": "proposed"
        },
        "prediction": {
            "method": "ALL"
        },
        "output": {
            "verbose": True,
            "save_visualizations": True
        }
    }
    
    @classmethod
    def load_config(cls, config_path="config.json"):
        """載入配置文件，如果不存在則創建默認配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"✓ 已載入配置文件: {config_path}")
        else:
            config = cls.DEFAULT_CONFIG.copy()
            cls.save_config(config, config_path)
            print(f"✓ 已創建默認配置文件: {config_path}")
        
        return cls.validate_config(config)
    
    @classmethod
    def save_config(cls, config, config_path="config.json"):
        """保存配置文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def validate_config(cls, config):
        """驗證並補充配置"""
        # 簡單的配置驗證
        if 'method' not in config or 'name' not in config['method']:
            config['method'] = cls.DEFAULT_CONFIG['method']
        
        # 確保預測方法映射正確
        method_name = config['method']['name']
        if method_name not in ['rotation', 'split', 'quadtree']:
            print(f"Warning: Unknown method '{method_name}', using 'quadtree'")
            config['method']['name'] = 'quadtree'
        
        return config

# =============================================================================
# 實驗執行器
# =============================================================================

class ExperimentRunner:
    """統一的實驗執行器"""
    
    def __init__(self, config):
        self.config = config
        self.viz_system = PEEVisualizationSystem()
        self.prediction_method_map = {
            "PROPOSED": PredictionMethod.PROPOSED,
            "MED": PredictionMethod.MED,
            "GAP": PredictionMethod.GAP,
            "RHOMBUS": PredictionMethod.RHOMBUS
        }
    
    def run_experiment(self):
        """執行完整實驗"""
        try:
            # 1. 檢查實驗類型
            if self.config['measurement']['use_method_comparison']:
                return self._run_method_comparison()
            elif self.config['prediction']['method'] == "ALL":
                return self._run_multi_predictor_experiment()
            else:
                return self._run_single_experiment()
                
        except Exception as e:
            print(f"❌ 實驗執行失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cleanup_memory()
    
    def _run_method_comparison(self):
        """執行方法比較"""
        config = self.config
        print(f"\n🔬 執行方法比較實驗")
        print(f"比較預測器: {config['measurement']['comparison_predictor']}")
        print(f"比較方法: {config['measurement']['methods_to_compare']}")
        
        results = run_method_comparison(
            imgName=config['image']['name'],
            filetype=config['image']['filetype'],
            predictor=config['measurement']['comparison_predictor'],
            ratio_of_ones=config['embedding']['predictor_ratios'].get(
                config['measurement']['comparison_predictor'].upper(), 0.5
            ),
            methods=config['measurement']['methods_to_compare'],
            method_params=self._build_method_params(),
            total_embeddings=config['embedding']['total_embeddings'],
            el_mode=config['embedding']['el_mode'],
            segments=config['measurement']['stats_segments'],
            step_size=config['measurement']['step_size']
        )
        
        print("✓ 方法比較完成")
        return results
    
    def _run_multi_predictor_experiment(self):
        """執行多預測器實驗"""
        config = self.config
        print(f"\n🔬 執行多預測器實驗")
        print(f"測試預測器: {list(config['embedding']['predictor_ratios'].keys())}")
        
        if config['measurement']['use_precise_measurement']:
            print("📊 使用精確測量模式")
            results = run_multi_predictor_precise_measurements(
                imgName=config['image']['name'],
                filetype=config['image']['filetype'],
                method=config['method']['name'],
                predictor_ratios=config['embedding']['predictor_ratios'],
                total_embeddings=config['embedding']['total_embeddings'],
                el_mode=config['embedding']['el_mode'],
                segments=config['measurement']['stats_segments'],
                step_size=config['measurement']['step_size'],
                use_different_weights=config['embedding']['use_different_weights'],
                split_size=config['method'].get('rotation', {}).get('split_size', 2),
                block_base=config['method'].get('split', {}).get('block_base', False),
                quad_tree_params=config['method'].get('quadtree', {})
            )
        else:
            print("📈 使用標準測量模式")
            # 調用標準的多預測器處理
            results = self._run_multi_predictor_standard()
        
        print("✓ 多預測器實驗完成")
        return results
    
    def _run_single_experiment(self):
        """執行單一實驗"""
        config = self.config
        prediction_method_str = config['prediction']['method']
        
        print(f"\n🔬 執行單一實驗")
        print(f"圖像: {config['image']['name']}.{config['image']['filetype']}")
        print(f"方法: {config['method']['name']}")
        print(f"預測器: {prediction_method_str}")
        
        # 讀取圖像
        img_path = find_image_path(config['image']['name'], config['image']['filetype'])
        orig_img, is_grayscale = self._read_image_auto(img_path)
        img_info = self.viz_system.get_image_info(orig_img)
        
        print(f"✓ 已載入圖像: {img_info['description']}")
        
        # 設置目錄
        directories = get_output_directories(config['image']['name'], config['method']['name'])
        create_all_directories(directories)
        
        # 獲取預測方法
        prediction_method = self.prediction_method_map.get(
            prediction_method_str.upper(), PredictionMethod.PROPOSED
        )
        ratio_of_ones = config['embedding']['predictor_ratios'].get(
            prediction_method_str.upper(), 0.5
        )
        
        # 檢查是否使用精確測量
        if config['measurement']['use_precise_measurement']:
            return self._run_precise_measurement(
                orig_img, prediction_method, prediction_method_str, ratio_of_ones
            )
        else:
            return self._run_standard_embedding(
                orig_img, img_info, directories, prediction_method, 
                prediction_method_str, ratio_of_ones, is_grayscale
            )
    
    def _run_precise_measurement(self, orig_img, prediction_method, method_str, ratio_of_ones):
        """執行精確測量"""
        config = self.config
        
        print("📊 使用精確測量模式")
        is_proposed = method_str.upper() == "PROPOSED"
        
        if is_proposed:
            results = run_precise_measurements(
                orig_img, config['image']['name'], config['method']['name'], 
                prediction_method, ratio_of_ones, config['embedding']['total_embeddings'], 
                config['embedding']['el_mode'], config['measurement']['stats_segments'], 
                config['measurement']['step_size'], config['embedding']['use_different_weights'],
                config['method'].get('rotation', {}).get('split_size', 2),
                config['method'].get('split', {}).get('block_base', False),
                config['method'].get('quadtree', {})
            )
        else:
            results = run_simplified_precise_measurements(
                orig_img, config['image']['name'], config['method']['name'], 
                prediction_method, ratio_of_ones, config['embedding']['total_embeddings'], 
                config['embedding']['el_mode'], config['measurement']['stats_segments'], 
                config['measurement']['step_size'], config['embedding']['use_different_weights'],
                config['method'].get('rotation', {}).get('split_size', 2),
                config['method'].get('split', {}).get('block_base', False),
                config['method'].get('quadtree', {})
            )
        
        return results
    
    def _run_standard_embedding(self, orig_img, img_info, directories, 
                               prediction_method, method_str, ratio_of_ones, is_grayscale):
        """執行標準嵌入處理"""
        config = self.config
        
        print("🔧 執行標準嵌入處理")
        
        # 保存原始圖像
        self.viz_system.save_image(orig_img, f"{directories['image']}/original.png")
        
        # 執行嵌入處理
        try:
            if config['method']['name'] == "rotation":
                final_img, total_payload, stages = pee_process_with_rotation_cuda(
                    orig_img, config['embedding']['total_embeddings'], ratio_of_ones,
                    config['embedding']['use_different_weights'],
                    config['method']['rotation']['split_size'],
                    config['embedding']['el_mode'], prediction_method=prediction_method
                )
            elif config['method']['name'] == "split":
                final_img, total_payload, stages = pee_process_with_split_cuda(
                    orig_img, config['embedding']['total_embeddings'], ratio_of_ones,
                    config['embedding']['use_different_weights'],
                    config['method']['split']['split_size'],
                    config['embedding']['el_mode'],
                    config['method']['split']['block_base'],
                    prediction_method=prediction_method
                )
            elif config['method']['name'] == "quadtree":
                final_img, total_payload, stages = pee_process_with_quadtree_cuda(
                    orig_img, config['embedding']['total_embeddings'], ratio_of_ones,
                    config['embedding']['use_different_weights'],
                    config['method']['quadtree']['min_block_size'],
                    config['method']['quadtree']['variance_threshold'],
                    config['embedding']['el_mode'],
                    rotation_mode='random',
                    prediction_method=prediction_method,
                    imgName=config['image']['name'],
                    output_dir="./Prediction_Error_Embedding/outcome",
                    adaptive_threshold=config['method']['quadtree'].get('adaptive_threshold', False),
                    search_mode=config['method']['quadtree'].get('search_mode', 'balanced'),
                    target_bpp_for_search=config['method']['quadtree'].get('target_bpp_for_search', 0.8),
                    target_psnr_for_search=config['method']['quadtree'].get('target_psnr_for_search', 35.0)
                )
            else:
                raise ValueError(f"Unknown method: {config['method']['name']}")
            
        except Exception as e:
            print(f"❌ 嵌入處理失敗: {str(e)}")
            raise
        
        # 計算最終指標
        final_metrics = self.viz_system.calculate_metrics(orig_img, final_img, img_info['is_color'])
        final_bpp = total_payload / img_info['pixel_count']
        
        # 保存最終結果
        self.viz_system.save_image(final_img, f"{directories['image']}/final_result.png")
        
        # 生成視覺化（僅針對PROPOSED預測器）
        if config['output']['save_visualizations'] and method_str.upper() == "PROPOSED":
            print("🎨 生成視覺化內容...")
            
            # 創建對比圖
            self.viz_system.save_comparison_image(
                orig_img, final_img, f"{directories['image']}/original_vs_final.png"
            )
            
            # 創建熱圖
            self.viz_system.create_heatmap(
                orig_img, final_img, f"{directories['image']}/final_heatmap.png"
            )
            
            # 創建直方圖對比
            self.viz_system.create_histogram_comparison(
                orig_img, final_img, f"{directories['histogram']}/histogram_comparison.png"
            )
        
        # 輸出結果摘要
        self._print_results_summary(
            img_info, total_payload, final_bpp, final_metrics, 
            config['method']['name'], method_str
        )
        
        return {
            'final_image': final_img,
            'total_payload': total_payload,
            'final_bpp': final_bpp,
            'stages': stages,
            'metrics': final_metrics
        }
    
    def _read_image_auto(self, filepath):
        """讀取圖像並自動判斷類型"""
        # 嘗試讀取為彩色圖像
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {filepath}")
        
        # 判斷是否為灰階圖像
        img_info = self.viz_system.get_image_info(img)
        is_grayscale = not img_info['is_color']
        
        if is_grayscale:
            # 轉換為灰階
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img, is_grayscale
        """構建方法參數字典"""
        config = self.config
        return {
            "rotation": {
                "split_size": config['method'].get('rotation', {}).get('split_size', 2),
                "use_different_weights": config['embedding']['use_different_weights']
            },
            "split": {
                "split_size": config['method'].get('split', {}).get('split_size', 2),
                "block_base": config['method'].get('split', {}).get('block_base', False),
                "use_different_weights": config['embedding']['use_different_weights']
            },
            "quadtree": {
                **config['method'].get('quadtree', {}),
                "use_different_weights": config['embedding']['use_different_weights']
            }
        }
    
    def _run_multi_predictor_standard(self):
        """執行標準的多預測器處理"""
        # 這裡可以實現標準模式的多預測器邏輯
        # 暫時返回None，表示使用精確測量模式
        print("⚠️  標準模式的多預測器處理尚未實現，請使用精確測量模式")
        return None
    
    def _print_results_summary(self, img_info, total_payload, final_bpp, metrics, method, predictor):
        """輸出結果摘要"""
        print(f"\n{'='*60}")
        print(f"🎯 實驗結果摘要")
        print(f"{'='*60}")
        print(f"圖像類型: {img_info['type_name'].capitalize()}")
        print(f"像素計數: {img_info['pixel_count']:,}")
        print(f"處理方法: {method}")
        print(f"預測器: {predictor}")
        print(f"總嵌入量: {total_payload:,} bits")
        print(f"最終BPP: {final_bpp:.6f}")
        print(f"最終PSNR: {metrics['psnr']:.2f} dB")
        print(f"最終SSIM: {metrics['ssim']:.4f}")
        print(f"直方圖相關性: {metrics['hist_corr']:.4f}")
        print(f"{'='*60}")

# =============================================================================
# 主程式
# =============================================================================

def main():
    """重構後的主程式 - 簡潔高效"""
    
    print("🚀 PEE 預測誤差嵌入系統 - 重構版")
    print("=" * 50)
    
    try:
        # 1. 載入配置
        config = ConfigManager.load_config()
        
        # 2. 創建並執行實驗
        runner = ExperimentRunner(config)
        results = runner.run_experiment()
        
        if results is not None:
            print("\n✅ 實驗執行成功！")
        else:
            print("\n❌ 實驗執行失敗！")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  用戶中斷實驗")
        return None
    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main()