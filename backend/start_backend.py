#!/root/anaconda3/envs/ocr_rag/bin/python
"""
Backend startup script for AgenticRAGOCR
使用 conda ocr_rag 环境启动后端服务
"""
import uvicorn
import os
import sys

# 打印 Python 信息，方便调试
print(f"Using Python: {sys.executable}")
print(f"Python version: {sys.version}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # 禁用自动重载
    )
