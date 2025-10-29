"""
Application Configuration
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # 阿里云百炼配置
    dashscope_api_key: str = ""
    qwen_model_name: str = "qwen-max"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2000

    # 向量数据库配置
    chroma_persist_dir: str = "./data/chroma_db"
    embedding_model: str = "text-embedding-v3"  # Qwen embedding 模型

    # RAG 配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # PaddleOCR 模型路径
    paddleocr_vl_model_dir: str = "/home/MuyuWorkSpace/02_OcrRag/PaddleOCR-VL-0.9B"
    layout_detection_model_dir: str = "/home/MuyuWorkSpace/02_OcrRag/PP-DocLayoutV2"

    # 文件上传配置
    upload_dir: str = "./uploads"
    max_upload_size: int = 50  # MB
    allowed_extensions: List[str] = ["pdf", "png", "jpg", "jpeg"]

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:5173"]

    # 日志配置
    log_level: str = "INFO"

    @field_validator('allowed_extensions', 'cors_origins', mode='before')
    @classmethod
    def parse_list_from_str(cls, v):
        """将逗号分隔的字符串转换为列表"""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保必要的目录存在
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


# 全局设置实例
settings = Settings()
