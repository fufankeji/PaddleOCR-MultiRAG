#!/bin/bash

echo "========================================="
echo "安装所有依赖"
echo "========================================="

# 创建虚拟环境
# 检查 Conda
if ! command -v conda &> /dev/null; then
    echo "未检测到 Conda，请先安装 Miniconda 或 Anaconda。"
    exit 1
fi

# 初始化 Conda
source $(conda info --base)/etc/profile.d/conda.sh

# 创建虚拟环境
echo  "创建 Conda 环境 ocr_rag"
conda create -n ocr_rag python=3.11 -y
conda activate ocr_rag

# 安装paddle依赖
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
python -m pip install "paddleocr[all]"

# 安装模型
echo "安装模型..."
pip install modelscope
python download_paddleocr_vl.py


# 安装后端依赖
echo "安装后端依赖..."
cd ./backend
pip install -r requirements.txt

# 安装前端依赖
echo "安装前端依赖..."
cd ../frontend
npm install


# 安装完成
echo "安装完成"
