"""
PaddleOCR Service for Document Parsing
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

from app.config import settings
from app.models.schemas import ParsedBlock, OCRResult, BlockType, DocumentStats

logger = logging.getLogger(__name__)

class OCRService:
    """PaddleOCR 文档解析服务"""

    def __init__(self):
        self.pipeline = None
        self._initialized = False

    async def initialize(self):
        """初始化 PaddleOCR pipeline"""
        if self._initialized:
            return

        try:
            logger.info("Initializing PaddleOCR pipeline...")

            # 在线程池中初始化（避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_pipeline)

            if self.pipeline is not None:
                self._initialized = True
                logger.info("PaddleOCR pipeline initialized successfully")
            else:
                raise Exception("Pipeline initialization returned None")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            logger.warning("PaddleOCR initialization failed, service will not be available")
            self._initialized = False
            self.pipeline = None

    def _init_pipeline(self):
        """初始化 pipeline（同步）"""
        try:
            import os
            # 设置使用 GPU 1（避免与 GPU 0 上的其他服务冲突）
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'

            from paddleocr import PaddleOCRVL

            self.pipeline = PaddleOCRVL(
                vl_rec_model_dir=settings.paddleocr_vl_model_dir,
                layout_detection_model_dir=settings.layout_detection_model_dir
            )
            logger.info(f"PaddleOCRVL loaded with model: {settings.paddleocr_vl_model_dir} on GPU 1")
        except Exception as e:
            logger.error(f"Error in _init_pipeline: {e}")
            self.pipeline = None
            raise

    async def parse_document(
        self,
        file_path: str,
        doc_id: str,
        output_dir: Optional[str] = None
    ) -> List[OCRResult]:
        """
        解析文档并返回结构化结果

        Args:
            file_path: 文档路径
            doc_id: 文档ID
            output_dir: 输出目录（保存JSON、Markdown等）

        Returns:
            每页的 OCR 结果列表
        """
        if not self._initialized:
            await self.initialize()

        # 检查 pipeline 是否可用
        if self.pipeline is None:
            raise Exception("PaddleOCR pipeline is not available. Please check the initialization logs.")

        try:
            start_time = time.time()

            # 准备输出目录
            if output_dir is None:
                output_dir = Path(settings.upload_dir) / doc_id
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 在线程池中执行 OCR（避免阻塞）
            loop = asyncio.get_event_loop()
            ocr_outputs = await loop.run_in_executor(
                None,
                self._run_ocr,
                file_path,
                str(output_dir)
            )

            # 从保存的JSON文件解析结果(而不是从内存对象)
            results = []
            json_files = sorted(output_dir.glob("*_res.json"))

            if json_files:
                # 从JSON文件读取
                import json
                for json_file in json_files:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    ocr_result = self._parse_json_output(json_data, doc_id)
                    results.append(ocr_result)
                    logger.info(f"Parsed {json_file.name}: {len(ocr_result.blocks)} blocks")
            else:
                # Fallback: 从内存对象解析
                for res in ocr_outputs:
                    ocr_result = self._parse_ocr_output(res, doc_id)
                    results.append(ocr_result)

            processing_time = time.time() - start_time
            logger.info(
                f"Document {doc_id} parsed: {len(results)} pages, "
                f"time: {processing_time:.2f}s"
            )

            return results

        except Exception as e:
            logger.error(f"Error parsing document {doc_id}: {e}")
            raise

    def _run_ocr(self, file_path: str, output_dir: str):
        """运行 OCR（同步）"""
        output = self.pipeline.predict(
            input=file_path,
            save_path=output_dir
        )

        # 保存结果到文件
        for res in output:
            res.save_to_json(save_path=output_dir)
            res.save_to_markdown(save_path=output_dir)
            res.save_to_img(save_path=output_dir)  # 保存可视化图片

        return output

    def _parse_json_output(self, json_data: dict, doc_id: str) -> OCRResult:
        """从JSON数据解析OCR结果"""
        # 确保page_index是整数
        page_index = json_data.get('page_index')
        if page_index is None:
            page_index = 0

        parsing_results = json_data.get('parsing_res_list', [])

        # 转换为 ParsedBlock
        blocks = []
        for item in parsing_results:
            try:
                # 确保block_order是整数或None
                block_order = item.get('block_order')
                if block_order is not None and not isinstance(block_order, int):
                    try:
                        block_order = int(block_order)
                    except:
                        block_order = None

                block = ParsedBlock(
                    block_id=item.get('block_id', 0),
                    block_label=item.get('block_label', 'text'),
                    block_content=item.get('block_content', ''),
                    block_bbox=item.get('block_bbox', [0, 0, 0, 0]),
                    block_order=block_order,
                    page_index=page_index
                )
                blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to parse block: {e}")
                continue

        return OCRResult(
            doc_id=doc_id,
            page_index=page_index,
            blocks=blocks,
            total_blocks=len(blocks),
            processing_time=0.0
        )

    def _parse_ocr_output(self, res, doc_id: str) -> OCRResult:
        """解析单页 OCR 输出(从内存对象)"""
        # 获取原始数据
        page_data = res.__dict__ if hasattr(res, '__dict__') else {}

        page_index = page_data.get('page_index', 0)
        parsing_results = page_data.get('parsing_res_list', [])

        # 转换为 ParsedBlock
        blocks = []
        for item in parsing_results:
            try:
                block = ParsedBlock(
                    block_id=item.get('block_id', 0),
                    block_label=item.get('block_label', 'text'),
                    block_content=item.get('block_content', ''),
                    block_bbox=item.get('block_bbox', [0, 0, 0, 0]),
                    block_order=item.get('block_order'),
                    page_index=page_index
                )
                blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to parse block: {e}")
                continue

        return OCRResult(
            doc_id=doc_id,
            page_index=page_index,
            blocks=blocks,
            total_blocks=len(blocks),
            processing_time=0.0
        )

    def calculate_stats(self, ocr_results: List[OCRResult]) -> DocumentStats:
        """计算文档统计信息"""
        stats = DocumentStats(
            doc_id=ocr_results[0].doc_id if ocr_results else "",
            text_blocks=0,
            table_blocks=0,
            image_blocks=0,
            formula_blocks=0,
            total_blocks=0
        )

        for result in ocr_results:
            for block in result.blocks:
                stats.total_blocks += 1

                label = block.block_label.lower()
                if 'table' in label:
                    stats.table_blocks += 1
                elif any(x in label for x in ['image', 'figure', 'chart']):
                    stats.image_blocks += 1
                elif 'formula' in label or 'equation' in label:
                    stats.formula_blocks += 1
                else:
                    stats.text_blocks += 1

        return stats

    def get_block_type(self, label: str) -> str:
        """获取块类型的统一标签"""
        label = label.lower()

        if 'table' in label:
            return 'table'
        elif any(x in label for x in ['image', 'figure', 'chart']):
            return 'image'
        elif 'formula' in label or 'equation' in label:
            return 'formula'
        else:
            return 'text'

    async def cleanup(self):
        """清理资源"""
        self.pipeline = None
        self._initialized = False
        logger.info("OCR service cleaned up")
