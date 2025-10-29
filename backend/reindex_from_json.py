"""
从已保存的JSON文件重新索引文档
用于修复OCR解析成功但向量索引失败的情况
"""

import json
import asyncio
import sys
from pathlib import Path

# 添加app目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag_service import RAGService
from app.models.schemas import ParsedBlock, OCRResult, DocumentStats
from app.config import settings

async def reindex_document(doc_id: str, json_path: str, file_name: str):
    """从JSON文件重新索引文档"""

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 解析blocks
    blocks = []
    for item in data['parsing_res_list']:
        block = ParsedBlock(
            block_id=item.get('block_id', 0),
            block_label=item.get('block_label', 'text'),
            block_content=item.get('block_content', ''),
            block_bbox=item.get('block_bbox', [0, 0, 0, 0]),
            block_order=item.get('block_order')
        )
        blocks.append(block)

    # 创建OCR结果
    ocr_result = OCRResult(
        doc_id=doc_id,
        page_index=0,  # 单页图片
        blocks=blocks,
        total_blocks=len(blocks),
        processing_time=0.0
    )

    print(f"从JSON读取到 {len(blocks)} 个块")

    # 初始化RAG服务
    rag_service = RAGService()
    await rag_service.initialize()

    # 重新索引
    print(f"开始重新索引文档 {doc_id}...")
    stats = await rag_service.index_document(
        doc_id=doc_id,
        file_name=file_name,
        ocr_results=[ocr_result]
    )

    print(f"索引完成！")
    print(f"  文本块: {stats.text_blocks}")
    print(f"  表格块: {stats.table_blocks}")
    print(f"  图像块: {stats.image_blocks}")
    print(f"  公式块: {stats.formula_blocks}")
    print(f"  总计: {stats.total_blocks}")

    return stats

async def main():
    # 最新的文档ID
    doc_id = "fee551f0-e8ad-4959-9a8f-541ef6f74f48"
    json_path = f"/home/MuyuWorkSpace/02_OcrRag/projects/AgenticRAGOCR/backend/uploads/{doc_id}/paddleocrvl_res.json"
    file_name = "paddleocrvl.png"

    if not Path(json_path).exists():
        print(f"错误: JSON文件不存在: {json_path}")
        return

    stats = await reindex_document(doc_id, json_path, file_name)

    # 更新进度信息
    print(f"\n现在可以通过以下API查看:")
    print(f"curl http://localhost:8000/api/progress/{doc_id}")

if __name__ == "__main__":
    asyncio.run(main())
