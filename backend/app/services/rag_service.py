"""
RAG Service for Document Indexing and Retrieval
"""

import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio

# Fix for sqlite3 missing in conda environment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dashscope
from dashscope import TextEmbedding

from app.config import settings
from app.models.schemas import OCRResult, ParsedBlock, Citation, DocumentStats

logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务 - 文档索引与检索"""

    def __init__(self):
        self.chroma_client = None
        self.text_splitter = None
        self._initialized = False
        self.collections = {}
        self.embedding_batch_size = 10  # Qwen API 批次大小限制（最大10）

    async def initialize(self):
        """初始化 RAG 服务"""
        if self._initialized:
            return

        try:
            logger.info("Initializing RAG service...")

            # 在线程池中初始化（避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_components)

            self._initialized = True
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise

    def _init_components(self):
        """初始化组件（同步）"""
        # 初始化 ChromaDB
        chroma_dir = Path(settings.chroma_persist_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 设置 DashScope API Key（用于 Embedding）
        dashscope.api_key = settings.dashscope_api_key
        logger.info(f"Using Qwen embedding model: {settings.embedding_model}")

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    async def index_document(
        self,
        doc_id: str,
        ocr_results: List[OCRResult],
        file_name: str
    ) -> DocumentStats:
        """
        索引文档到向量数据库

        Args:
            doc_id: 文档ID
            ocr_results: OCR 解析结果
            file_name: 原始文件名

        Returns:
            文档统计信息
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"Indexing document {doc_id}...")

            # 创建文档专属的 collection
            collection_name = f"doc_{doc_id}"

            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None,
                self._get_or_create_collection,
                collection_name
            )

            self.collections[doc_id] = collection

            # 准备索引数据
            all_chunks = []
            all_embeddings = []
            all_metadatas = []
            all_ids = []

            chunk_id = 0
            stats = DocumentStats(
                doc_id=doc_id,
                text_blocks=0,
                table_blocks=0,
                image_blocks=0,
                formula_blocks=0,
                total_blocks=0
            )

            for ocr_result in ocr_results:
                page_index = ocr_result.page_index

                for block in ocr_result.blocks:
                    stats.total_blocks += 1

                    # 统计块类型
                    block_type = self._get_block_type(block.block_label)
                    if block_type == "text":
                        stats.text_blocks += 1
                    elif block_type == "table":
                        stats.table_blocks += 1
                    elif block_type == "image":
                        stats.image_blocks += 1
                    elif block_type == "formula":
                        stats.formula_blocks += 1

                    # 准备块的元数据
                    base_metadata = {
                        "doc_id": doc_id,
                        "file_name": file_name,
                        "page_index": page_index,
                        "block_id": block.block_id,
                        "block_type": block_type,
                        "block_label": block.block_label,
                        "block_bbox": json.dumps(block.block_bbox),
                        "block_order": block.block_order if block.block_order is not None else 0
                    }

                    content = block.block_content.strip()
                    if not content:
                        continue

                    # 对于长文本块，进行分块
                    if block_type == "text" and len(content) > settings.chunk_size:
                        chunks = self.text_splitter.split_text(content)

                        for i, chunk in enumerate(chunks):
                            metadata = base_metadata.copy()
                            metadata["chunk_index"] = i
                            metadata["total_chunks"] = len(chunks)
                            metadata["is_chunked"] = True

                            all_chunks.append(chunk)
                            all_metadatas.append(metadata)
                            all_ids.append(f"{doc_id}_p{page_index}_b{block.block_id}_c{i}")
                            chunk_id += 1
                    else:
                        # 表格、图片、公式等不分块
                        metadata = base_metadata.copy()
                        metadata["is_chunked"] = False

                        all_chunks.append(content)
                        all_metadatas.append(metadata)
                        all_ids.append(f"{doc_id}_p{page_index}_b{block.block_id}")
                        chunk_id += 1

            # 批量生成 embeddings
            if all_chunks:
                logger.info(f"Generating embeddings for {len(all_chunks)} chunks using Qwen...")
                all_embeddings = await self._generate_embeddings(all_chunks)

                # 批量插入到 ChromaDB
                logger.info(f"Inserting {len(all_chunks)} chunks into ChromaDB...")
                await loop.run_in_executor(
                    None,
                    collection.add,
                    all_ids,
                    all_embeddings,
                    all_metadatas,
                    all_chunks
                )

            logger.info(f"Document {doc_id} indexed successfully: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            raise

    async def query(
        self,
        doc_id: str,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        查询文档

        Args:
            doc_id: 文档ID
            query_text: 查询文本
            top_k: 返回结果数量

        Returns:
            检索结果列表，包含内容和元数据
        """
        if not self._initialized:
            await self.initialize()

        try:
            collection = self.collections.get(doc_id)
            if not collection:
                collection_name = f"doc_{doc_id}"
                try:
                    loop = asyncio.get_event_loop()
                    collection = await loop.run_in_executor(
                        None,
                        self.chroma_client.get_collection,
                        collection_name
                    )
                    self.collections[doc_id] = collection
                except Exception:
                    logger.warning(f"Collection for doc {doc_id} not found")
                    return []

            # 生成查询 embedding
            query_embeddings = await self._generate_embeddings([query_text])
            query_embedding = query_embeddings[0]

            # 查询 ChromaDB
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            )

            # 转换结果格式
            retrieved_results = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    result_item = {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1 - results["distances"][0][i],  # 距离转相似度
                    }
                    retrieved_results.append(result_item)

            return retrieved_results

        except Exception as e:
            logger.error(f"Error querying document {doc_id}: {e}")
            raise

    def _get_or_create_collection(self, collection_name: str):
        """获取或创建 collection"""
        try:
            collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists, clearing...")
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return collection

    def _get_block_type(self, label: str) -> str:
        """获取块类型的统一标签"""
        label = label.lower()

        if "table" in label:
            return "table"
        elif "formula" in label or "equation" in label:
            return "formula"
        # 区分图像本身和图像的标题/说明文字
        elif "title" in label or "caption" in label or "footnote" in label:
            return "text"  # 标题、说明、脚注都是文本
        elif any(x in label for x in ["image", "figure", "chart"]):
            return "image"
        else:
            return "text"

    def format_citations(self, results: List[Dict[str, Any]]) -> List[Citation]:
        """
        将检索结果转换为前端引用格式

        Args:
            results: 检索结果

        Returns:
            引用列表
        """
        citations = []
        for i, result in enumerate(results):
            metadata = result["metadata"]
            content = result["content"]
            block_type = metadata.get("block_type", "text")

            try:
                bbox = json.loads(metadata.get("block_bbox", "[]"))
            except:
                bbox = None

            # 对于图片和表格，构建图片路径
            image_path = None
            block_label = metadata.get("block_label", "")
            doc_id = metadata.get("doc_id")

            # 检查是否有对应的图片文件（image类型的block都会有图片）
            if bbox and len(bbox) == 4 and block_label.lower() == "image":
                # 图片命名格式：img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg
                image_filename = f"img_in_image_box_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"
                image_path = f"/api/documents/{doc_id}/images/{image_filename}"

            citation = Citation(
                id=i + 1,
                source=metadata.get("file_name", ""),
                page=metadata.get("page_index", 0),
                snippet=content[:200],  # 截取前200字符作为预览
                content=content,  # 完整内容
                type=block_type,
                block_id=metadata.get("block_id"),
                bbox=bbox,
                image_path=image_path,
                score=result.get("score", 0.0)
            )
            citations.append(citation)

        return citations

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        使用 Qwen Embedding API 生成向量

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        all_embeddings = []

        # 分批处理（Qwen API 有批次大小限制）
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i:i + self.embedding_batch_size]

            try:
                # 在线程池中调用 API（避免阻塞）
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self._call_embedding_api,
                    batch
                )

                # 提取 embeddings
                if response.status_code == 200:
                    embeddings = [item['embedding'] for item in response.output['embeddings']]
                    all_embeddings.extend(embeddings)
                    logger.debug(f"Generated embeddings for batch {i//self.embedding_batch_size + 1}")
                else:
                    error_msg = f"Embedding API error: {response.code} - {response.message}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//self.embedding_batch_size + 1}: {e}")
                raise

        return all_embeddings

    def _call_embedding_api(self, texts: List[str]):
        """同步调用 Embedding API"""
        response = TextEmbedding.call(
            model=settings.embedding_model,
            input=texts
        )
        return response

    async def delete_document(self, doc_id: str):
        """删除文档索引"""
        try:
            collection_name = f"doc_{doc_id}"
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.chroma_client.delete_collection,
                collection_name
            )

            if doc_id in self.collections:
                del self.collections[doc_id]

            logger.info(f"Document {doc_id} deleted from index")
        except Exception as e:
            logger.warning(f"Error deleting document {doc_id}: {e}")

    async def cleanup(self):
        """清理资源"""
        self.collections.clear()
        self.chroma_client = None
        self._initialized = False
        logger.info("RAG service cleaned up")
