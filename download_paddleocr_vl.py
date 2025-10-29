#模型下载
import os
import shutil
from modelscope import snapshot_download

# 下载模型到./model目录
model_dir = snapshot_download('PaddlePaddle/PaddleOCR-VL',local_dir='./model')

# 新建./model/PaddleOCR-VL-0.9B目录
target_dir = './model/PaddleOCR-VL-0.9B'
os.makedirs(target_dir, exist_ok=True)

# 把./model下所有文件迁移到./model/PaddleOCR-VL-0.9B内，文件夹除外
source_dir = './model'
for item in os.listdir(source_dir):
    item_path = os.path.join(source_dir, item)
    # 只移动文件，不移动文件夹
    if os.path.isfile(item_path):
        target_path = os.path.join(target_dir, item)
        shutil.move(item_path, target_path)
        print(f"已移动文件: {item} -> {target_dir}")

print(f"模型下载完成，文件已整理到: {target_dir}")