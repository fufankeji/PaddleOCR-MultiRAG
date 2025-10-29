"""
Data Models and Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class BlockType(str, Enum):
    """PaddleOCR 识别的块类型"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"
    CHART = "chart"
    FORMULA = "formula"
    PARAGRAPH_TITLE = "paragraph_title"
    DOC_TITLE = "doc_title"
    HEADER = "header"
    FOOTER = "footer"
    ABSTRACT = "abstract"
    ASIDE_TEXT = "aside_text"
    UNKNOWN = "unknown"

class ParsedBlock(BaseModel):
    """解析的文档块"""
    block_id: int
    block_label: str
    block_content: str
    block_bbox: List[int]  # [x1, y1, x2, y2]
    block_order: Optional[int] = None
    page_index: int = 0

class DocumentMetadata(BaseModel):
    """文档元数据"""
    doc_id: str
    filename: str
    file_type: str
    page_count: int
    upload_time: datetime
    file_size: int  # bytes
    ocr_status: str = "pending"  # pending, processing, completed, failed

class OCRResult(BaseModel):
    """OCR 处理结果"""
    doc_id: str
    page_index: int
    blocks: List[ParsedBlock]
    total_blocks: int
    processing_time: float

class DocumentStats(BaseModel):
    """文档统计信息"""
    doc_id: str
    text_blocks: int = 0
    table_blocks: int = 0
    image_blocks: int = 0
    formula_blocks: int = 0
    total_blocks: int = 0

class Citation(BaseModel):
    """引用信息"""
    id: int
    source: str
    page: int
    snippet: str
    content: Optional[str] = None  # 完整内容
    type: str  # text, table, image
    block_id: Optional[int] = None
    bbox: Optional[List[int]] = None
    image_path: Optional[str] = None  # 图片路径（用于image和table类型）
    score: float = 0.0  # 相关性分数

class ChatMessage(BaseModel):
    """聊天消息"""
    content: str
    is_user: bool
    citations: Optional[List[Citation]] = []
    timestamp: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    doc_id: Optional[str] = None  # 如果指定，只在该文档中查询
    top_k: int = 5  # 返回前k个相关块

class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    citations: List[Citation]
    doc_id: str

class UploadResponse(BaseModel):
    """文件上传响应"""
    doc_id: str
    file_name: str
    status: str
    message: str

class IndexingProgress(BaseModel):
    """索引构建进度"""
    doc_id: str
    status: str  # processing, completed, failed
    progress: int  # 0-100
    message: str
    stats: Optional[DocumentStats] = None

class DocumentPreview(BaseModel):
    """文档预览"""
    doc_id: str
    filename: str
    page_count: int
    preview_url: Optional[str] = None
    stats: DocumentStats
