<div align="center">
  <h1>🚀 PaddleOCR Traceable Multimodal RAG System</h1>
  <p><em>Powered by PaddleOCR-VL | Every answer has its source</em></p>
  <span>English | <a href="./README_zh.md">中文</a> </span>
</div>

## ⚡ Project Overview

This project is a traceable multimodal RAG system built on **PaddleOCR-VL**. It adopts a FastAPI backend + React frontend architecture.

![Project Image](assets/项目图片.png)

The system can efficiently process PDF and image files, with powerful OCR capabilities, supporting complex layout understanding such as chart parsing and formula recognition, and providing RAG traceable citation capabilities.

## 🎯 Key Features

### 🔍 Multimodal Precise Parsing
- Support for automatic parsing of complex architecture diagrams and engineering drawings
- Precise recognition of handwritten drawings and scanned document elements
- Classified parsing of PDF images and formulas
- Recognition results with coordinate information

### 📚 RAG Indexing
- Online index construction for large and complex PDFs
- Independent indexing of plain text blocks, LaTeX complex formulas, and HTML table data
- Independent splitting of various elements in images

### 💬 Q&A and Traceability
- Support for multi-turn Q&A in multimodal RAG knowledge base
- Coordinate-level fine-grained Q&A traceability display

### 🎨 Visualization and Interaction
- Real-time frontend updates of image parsing progress
- Dynamic rendering of document parsing structure
- Online preview of text-image layout recognition results

## 👀 Project Demo

![Demo Video](assets/演示视频.gif)

## 🚀 Usage Guide

### System Requirements

- **Operating System**: Requires running on Linux system
- **Python**: Recommended version 3.11
- **Node.js**: ≥ 18.0
- **GPU**: Recommended 8GB+; CUDA 12.6

### Quick Start
#### Method 1: One-Click Script Installation (Recommended)
Replace the `DASHSCOPE_API_KEY` in `backend/.env` with your own API key.
```bash
bash install_all.sh
bash start_all.sh
```
#### Method 2: Manual Installation

##### 1. Environment Preparation

Ensure the following software is installed on your system:
- **Conda**: Miniconda or Anaconda
- **Node.js**: ≥ 18.0
- **CUDA**: 12.6 (for GPU acceleration)

##### 2. Create Python Environment

```bash
# Create Conda virtual environment
conda create -n ocr_rag python=3.11 -y

# Activate environment
conda activate ocr_rag
```

##### 3. Install PaddlePaddle and OCR Dependencies

```bash
# Install PaddlePaddle GPU version (CUDA 12.6)
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install safetensors
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# Install PaddleOCR full version
python -m pip install "paddleocr[all]"
```

##### 4. Download PaddleOCR-VL Model

```bash
# Install ModelScope
pip install modelscope

# Download model files
python download_paddleocr_vl.py
```

##### 5. Install Backend Dependencies

```bash
# Enter backend directory
cd ./backend

# Install Python dependencies
pip install -r requirements.txt
```

##### 6. Install Frontend Dependencies

```bash
# Enter frontend directory
cd ../frontend

# Install Node.js dependencies
npm install
```

##### 7. Start Services

**Start Backend Service:**
```bash
# In backend directory
python start_backend.py
```
Backend will start at `http://localhost:8000`

**Start Frontend Service:**
```bash
# In frontend directory, open a new terminal
PORT=3001 npm run dev
```
Frontend will start at `http://localhost:3001`

## 🙈 Contributing
We welcome contributions to the project through GitHub PR submissions or issues. We greatly appreciate any form of contribution, including feature improvements, bug fixes, or documentation optimization.

## 😎 Technical Communication
Explore our technical community 👉 [AI Tech Community | Normed Space](https://kq4b3vgg5b.feishu.cn/wiki/JuJSwfbwmiwvbqkiQ7LcN1N1nhd)

Scan to add our assistant, reply "PaddleOCR-RAG" to join the technical exchange group and learn with other developers.
<div align="center">
<img src="assets\交流群.jpg" width="200" alt="Technical Exchange Group QR Code">
<div>