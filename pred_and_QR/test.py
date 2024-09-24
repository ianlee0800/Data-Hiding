import unittest
import numpy as np
import cupy as cp
import itertools
from pee import pee_embedding_adaptive_cuda
from image_processing import improved_predict_image_cuda
from common import calculate_psnr
from utils import brute_force_weight_search_cuda  # 替換 'your_module' 為實際的模塊名

class TestBruteForceWeightSearchCUDA(unittest.TestCase):
    def setUp(self):
        # 設置測試數據
        self.img_size = (64, 64)  # 使用小圖像以加快測試速度
        self.img = cp.random.randint(0, 256, self.img_size, dtype=cp.uint8)
        self.data = cp.random.randint(0, 2, self.img_size[0] * self.img_size[1], dtype=cp.uint8)
        self.EL = 3
        self.weight_range = range(1, 5)  # 縮小權重範圍以加快測試

    def test_brute_force_search(self):
        best_weights, (best_payload, best_psnr) = brute_force_weight_search_cuda(
            self.img, self.data, self.EL, self.weight_range
        )

        # 檢查返回值的類型和範圍
        self.assertIsInstance(best_weights, tuple)
        self.assertEqual(len(best_weights), 4)
        for w in best_weights:
            self.assertIn(w, self.weight_range)

        self.assertIsInstance(best_payload, (int, np.integer))
        self.assertGreater(best_payload, 0)
        self.assertIsInstance(best_psnr, float)
        self.assertGreater(best_psnr, 0)

        # 驗證最佳權重確實產生最佳結果
        for w1, w2, w3, w4 in itertools.product(self.weight_range, repeat=4):
            weights = (w1, w2, w3, w4)
            pred_img = improved_predict_image_cuda(self.img, weights)
            embedded, payload, _ = pee_embedding_adaptive_cuda(self.img, self.data, pred_img, self.EL)
            psnr = calculate_psnr(cp.asnumpy(self.img), cp.asnumpy(embedded))

            self.assertLessEqual(payload, best_payload)
            if payload == best_payload:
                self.assertLessEqual(psnr, best_psnr)

    def test_performance(self):
        import time

        start_time = time.time()
        brute_force_weight_search_cuda(self.img, self.data, self.EL, self.weight_range)
        end_time = time.time()

        print(f"CUDA Brute Force Search took {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    unittest.main()