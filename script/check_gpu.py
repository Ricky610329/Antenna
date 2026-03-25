import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ? python -m script.check_gpu

import time

import torch

# 檢查是否有可用的 GPU
if not torch.cuda.is_available():
    print("No GPU found. This script requires a CUDA-enabled GPU to run.")
    # 如果沒有 GPU，程式將停止
    exit()

# 將裝置設置為 GPU
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# 創建兩個非常大的矩陣
# 尺寸可以根據你的 GPU 記憶體大小調整，過大可能會導致記憶體不足 (OOM)
matrix_size = 10000  # 10000x10000 的矩陣
try:
    # 創建兩個隨機的 Tensor 並將它們移動到 GPU
    A = torch.randn(matrix_size, matrix_size, device=device)
    B = torch.randn(matrix_size, matrix_size, device=device)
    print(f"Created two {matrix_size}x{matrix_size} matrices on GPU.")

    print("Starting GPU stress test... Press Ctrl+C to stop.")

    start_time = time.time()
    iterations = 0

    # 無限迴圈，持續進行矩陣乘法
    while True:
        # 在 GPU 上執行矩陣乘法
        C = torch.matmul(A, B)
        iterations += 1

        # 每隔 10 秒印出一次進度
        if time.time() - start_time > 10:
            print(f"Completed {iterations} matmul operations in the last 10 seconds.")
            start_time = time.time()
            iterations = 0

except torch.cuda.OutOfMemoryError:
    print("\nCUDA Out of Memory Error! Please try reducing the 'matrix_size'.")
except KeyboardInterrupt:
    print("\nGPU stress test stopped by user.")
except Exception as e:
    print(f"An error occurred: {e}")
