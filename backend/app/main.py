"""
FastAPI Main Application
"""

import logging
import asyncio
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from app.config import settings
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IndexingProgress,
    DocumentStats,
    UploadResponse
)
from app.services.ocr_service import OCRService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局服务实例
ocr_service = OCRService()
rag_service = RAGService()
llm_service = LLMService()

# 文档处理进度跟踪
indexing_progress: dict[str, IndexingProgress] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    logger.info("Starting up services...")
    try:
        await ocr_service.initialize()
        await rag_service.initialize()
        await llm_service.initialize()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # 关闭时清理资源
    logger.info("Shutting down services...")
    await ocr_service.cleanup()
    await rag_service.cleanup()
    await llm_service.cleanup()
    logger.info("All services cleaned up")


# 创建 FastAPI 应用
app = FastAPI(
    title="Agentic RAG OCR API",
    description="Multimodal RAG system with PaddleOCR and Qwen",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "healthy",
        "message": "Agentic RAG OCR API is running",
        "version": "1.0.0"
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    上传文档并启动索引流程

    支持的格式: PDF, PNG, JPG, JPEG
    """
    try:
        # 验证文件类型
        allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )

        # 生成文档 ID
        doc_id = str(uuid.uuid4())

        # 保存上传的文件
        upload_dir = Path(settings.upload_dir) / doc_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"File uploaded: {file.filename} -> {doc_id}")

        # 初始化进度跟踪
        indexing_progress[doc_id] = IndexingProgress(
            doc_id=doc_id,
            status="processing",
            progress=0,
            message="文档上传成功，开始解析..."
        )

        # 在后台启动索引流程
        background_tasks.add_task(
            process_document,
            doc_id=doc_id,
            file_path=str(file_path),
            file_name=file.filename
        )

        return UploadResponse(
            doc_id=doc_id,
            file_name=file.filename,
            status="processing",
            message="文档上传成功，正在处理中..."
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document(doc_id: str, file_path: str, file_name: str):
    """
    后台任务：处理文档（OCR + 索引）

    Args:
        doc_id: 文档ID
        file_path: 文件路径
        file_name: 文件名
    """
    try:
        # 步骤1: OCR 解析
        indexing_progress[doc_id].progress = 10
        indexing_progress[doc_id].message = "正在进行 OCR 解析..."

        output_dir = Path(settings.upload_dir) / doc_id
        ocr_results = await ocr_service.parse_document(
            file_path=file_path,
            doc_id=doc_id,
            output_dir=str(output_dir)
        )

        logger.info(f"OCR completed for {doc_id}: {len(ocr_results)} pages")

        # 步骤2: 计算统计信息
        indexing_progress[doc_id].progress = 40
        indexing_progress[doc_id].message = "OCR 解析完成，正在分析文档结构..."

        stats = ocr_service.calculate_stats(ocr_results)
        logger.info(f"Stats from OCR: text={stats.text_blocks}, table={stats.table_blocks}, image={stats.image_blocks}, formula={stats.formula_blocks}, total={stats.total_blocks}")

        # 步骤3: 向量化索引
        indexing_progress[doc_id].progress = 50
        indexing_progress[doc_id].message = "正在构建向量索引..."

        stats = await rag_service.index_document(
            doc_id=doc_id,
            ocr_results=ocr_results,
            file_name=file_name
        )
        logger.info(f"Stats from RAG: text={stats.text_blocks}, table={stats.table_blocks}, image={stats.image_blocks}, formula={stats.formula_blocks}, total={stats.total_blocks}")

        # 完成
        indexing_progress[doc_id].status = "completed"
        indexing_progress[doc_id].progress = 100
        indexing_progress[doc_id].message = "索引构建完成！"
        indexing_progress[doc_id].stats = stats

        logger.info(f"Document {doc_id} processed successfully with stats: {stats}")

    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
        indexing_progress[doc_id].status = "failed"
        indexing_progress[doc_id].message = f"处理失败: {str(e)}"


@app.get("/api/progress/{doc_id}", response_model=IndexingProgress)
async def get_indexing_progress(doc_id: str):
    """
    获取文档索引进度

    Args:
        doc_id: 文档ID

    Returns:
        索引进度信息
    """
    if doc_id in indexing_progress:
        return indexing_progress[doc_id]

    # 如果内存中没有，尝试从OCR结果JSON文件读取统计信息
    try:
        import json
        from pathlib import Path

        # 查找OCR结果JSON文件
        doc_dir = Path(settings.upload_dir) / doc_id
        json_files = list(doc_dir.glob("*_res.json"))

        if not json_files:
            raise HTTPException(status_code=404, detail="Document not found")

        # 读取JSON文件
        with open(json_files[0], 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        # 从parsing_res_list统计
        blocks = ocr_data.get('parsing_res_list', [])

        text_count = 0
        table_count = 0
        image_count = 0
        formula_count = 0

        for block in blocks:
            label = block.get('block_label', '').lower()
            if 'table' in label:
                table_count += 1
            elif 'image' in label or 'figure' in label or 'chart' in label:
                image_count += 1
            elif 'formula' in label or 'equation' in label:
                formula_count += 1
            else:
                text_count += 1

        stats = DocumentStats(
            doc_id=doc_id,
            text_blocks=text_count,
            table_blocks=table_count,
            image_blocks=image_count,
            formula_blocks=formula_count,
            total_blocks=len(blocks)
        )

        return IndexingProgress(
            doc_id=doc_id,
            status="completed",
            progress=100,
            message="文档解析完成！",
            stats=stats
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving OCR results for {doc_id}: {e}")
        raise HTTPException(status_code=404, detail="Document not found")


@app.get("/api/documents/{doc_id}/blocks")
async def get_document_blocks(
    doc_id: str, 
    block_type: Optional[str] = None,
    page: Optional[int] = None
):
    """
    获取文档的所有blocks内容

    Args:
        doc_id: 文档ID
        block_type: 可选,筛选特定类型的块 (text, table, image, formula)
        page: 可选,筛选特定页码的块 (从0开始)

    Returns:
        文档blocks列表
    """
    try:
        import json
        from pathlib import Path

        # 查找OCR结果JSON文件
        doc_dir = Path(settings.upload_dir) / doc_id
        json_files = list(doc_dir.glob("*_res.json"))

        if not json_files:
            raise HTTPException(status_code=404, detail="Document not found")

        # 读取JSON文件并添加图片路径
        all_blocks = []
        imgs_dir = doc_dir / "imgs"

        # 按文件名排序，确保页码顺序
        json_files = sorted(json_files)

        for file_index, json_file in enumerate(json_files):
            # 如果指定了page参数，只处理对应页
            if page is not None and file_index != page:
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            blocks = ocr_data.get('parsing_res_list', [])
            page_index = ocr_data.get('page_index', file_index)  # 从JSON读取或使用文件索引

            # 为每个block添加页码和图片路径
            for block in blocks:
                # 添加页码信息
                block['page_index'] = page_index
                block['source_file'] = json_file.name

                label = block.get('block_label', '').lower()
                if 'image' in label or 'figure' in label or 'chart' in label:
                    # 根据bbox查找对应的图片文件
                    bbox = block.get('block_bbox', [])
                    if bbox and len(bbox) >= 4:
                        # 图片文件名格式: img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg
                        img_pattern = f"img_in_image_box_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"
                        img_file = imgs_dir / img_pattern
                        if img_file.exists():
                            block['image_path'] = f"/api/documents/{doc_id}/images/{img_pattern}"

                all_blocks.append(block)

        # 如果指定了类型,进行筛选(与rag_service._get_block_type保持一致)
        if block_type:
            filtered_blocks = []
            for block in all_blocks:
                label = block.get('block_label', '').lower()

                if block_type == 'text':
                    # 文本类型: 排除table、formula、纯image(但包含title/caption/footnote)
                    if 'table' in label:
                        continue
                    if 'formula' in label or 'equation' in label:
                        continue
                    # title、caption、footnote都是文本
                    if 'title' in label or 'caption' in label or 'footnote' in label:
                        filtered_blocks.append(block)
                    # 纯image/figure/chart不是文本
                    elif not any(x in label for x in ['image', 'figure', 'chart']):
                        filtered_blocks.append(block)
                elif block_type == 'table':
                    if 'table' in label:
                        filtered_blocks.append(block)
                elif block_type == 'image':
                    # 图像类型: 只包含纯image/figure/chart,排除title/caption/footnote
                    if 'title' in label or 'caption' in label or 'footnote' in label:
                        continue
                    if any(x in label for x in ['image', 'figure', 'chart']):
                        filtered_blocks.append(block)
                elif block_type == 'formula':
                    if 'formula' in label or 'equation' in label:
                        filtered_blocks.append(block)

            return {"blocks": filtered_blocks, "total": len(filtered_blocks)}

        return {"blocks": all_blocks, "total": len(all_blocks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving blocks for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/images/{image_name}")
async def get_document_image(doc_id: str, image_name: str):
    """
    获取文档中的图片文件

    Args:
        doc_id: 文档ID
        image_name: 图片文件名

    Returns:
        图片文件
    """
    try:
        from pathlib import Path
        from fastapi.responses import FileResponse

        # 构建图片路径
        img_path = Path(settings.upload_dir) / doc_id / "imgs" / image_name

        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        # 返回图片文件
        return FileResponse(img_path, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image {image_name} for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/visualizations")
async def get_document_visualizations(doc_id: str):
    """
    获取文档的可视化图片列表

    Args:
        doc_id: 文档ID

    Returns:
        原图和OCR可视化图片的信息
    """
    try:
        from pathlib import Path
        import mimetypes

        doc_dir = Path(settings.upload_dir) / doc_id

        if not doc_dir.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        visualizations = {
            "original": [],      # 原始文件
            "layout_det": [],    # 布局检测结果图
            "layout_order": [],  # 阅读顺序图
        }

        # 查找原始图片/PDF文件
        for ext in ['.png', '.jpg', '.jpeg', '.pdf']:
            original_files = list(doc_dir.glob(f"*{ext}"))
            for f in original_files:
                if not any(x in f.name for x in ['_layout_det_res', '_layout_order_res', '_res']):
                    mime_type, _ = mimetypes.guess_type(str(f))
                    visualizations["original"].append({
                        "name": f.name,
                        "path": f"/api/documents/{doc_id}/files/{f.name}",
                        "type": mime_type or "application/octet-stream"
                    })

        # 查找布局检测结果图
        layout_det_files = list(doc_dir.glob("*_layout_det_res.png"))
        for f in layout_det_files:
            visualizations["layout_det"].append({
                "name": f.name,
                "path": f"/api/documents/{doc_id}/files/{f.name}",
            })

        # 查找阅读顺序图
        layout_order_files = list(doc_dir.glob("*_layout_order_res.png"))
        for f in layout_order_files:
            visualizations["layout_order"].append({
                "name": f.name,
                "path": f"/api/documents/{doc_id}/files/{f.name}",
            })

        return visualizations

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving visualizations for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/files/{file_name}")
async def get_document_file(doc_id: str, file_name: str):
    """
    获取文档目录下的任意文件

    Args:
        doc_id: 文档ID
        file_name: 文件名

    Returns:
        文件内容
    """
    try:
        from pathlib import Path
        from fastapi.responses import FileResponse
        import mimetypes

        # 构建文件路径
        file_path = Path(settings.upload_dir) / doc_id / file_name

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # 确保文件在doc_id目录下(安全检查)
        if not str(file_path.resolve()).startswith(str((Path(settings.upload_dir) / doc_id).resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        # 猜测MIME类型
        mime_type, _ = mimetypes.guess_type(str(file_path))

        # 返回文件
        return FileResponse(file_path, media_type=mime_type or "application/octet-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file {file_name} for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    查询文档

    Args:
        request: 查询请求（包含 doc_id 和 query）

    Returns:
        查询响应（包含答案和引用）
    """
    try:
        # 检查文档是否已索引（先查内存，再查ChromaDB）
        is_ready = False

        if request.doc_id in indexing_progress:
            progress = indexing_progress[request.doc_id]
            if progress.status == "completed":
                is_ready = True
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document is not ready. Current status: {progress.status}"
                )
        else:
            # 内存中没有，尝试从ChromaDB验证
            try:
                collection_name = f"doc_{request.doc_id}"
                collection = rag_service.chroma_client.get_collection(name=collection_name)
                # 如果collection存在且有数据，说明已索引完成
                count = collection.count()
                if count > 0:
                    is_ready = True
                    logger.info(f"Document {request.doc_id} found in ChromaDB with {count} chunks")
                else:
                    raise HTTPException(status_code=404, detail="Document not found or empty")
            except Exception as e:
                logger.error(f"Document {request.doc_id} not found: {e}")
                raise HTTPException(status_code=404, detail="Document not found")

        if not is_ready:
            raise HTTPException(status_code=400, detail="Document is not ready")

        # 步骤1: 检索相关内容
        logger.info(f"Querying document {request.doc_id}: {request.query}")

        results = await rag_service.query(
            doc_id=request.doc_id,
            query_text=request.query,
            top_k=settings.retrieval_top_k
        )

        if not results:
            return QueryResponse(
                answer="抱歉，我在文档中没有找到相关信息。",
                citations=[],
                doc_id=request.doc_id
            )

        # 步骤2: 转换为引用格式
        citations = rag_service.format_citations(results)

        # 步骤3: 生成回答
        answer = await llm_service.generate_answer(
            query=request.query,
            context_chunks=results,
            citations=citations
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            doc_id=request.doc_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_document_stream(request: QueryRequest):
    """
    流式查询文档（SSE）

    Args:
        request: 查询请求

    Returns:
        流式响应
    """
    try:
        # 检查文档是否已索引（先查内存，再查ChromaDB）
        is_ready = False

        if request.doc_id in indexing_progress:
            progress = indexing_progress[request.doc_id]
            if progress.status == "completed":
                is_ready = True
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document is not ready. Current status: {progress.status}"
                )
        else:
            # 内存中没有，尝试从ChromaDB验证
            try:
                collection_name = f"doc_{request.doc_id}"
                collection = rag_service.chroma_client.get_collection(name=collection_name)
                count = collection.count()
                if count > 0:
                    is_ready = True
                    logger.info(f"Document {request.doc_id} found in ChromaDB with {count} chunks")
                else:
                    raise HTTPException(status_code=404, detail="Document not found or empty")
            except Exception as e:
                logger.error(f"Document {request.doc_id} not found: {e}")
                raise HTTPException(status_code=404, detail="Document not found")

        if not is_ready:
            raise HTTPException(status_code=400, detail="Document is not ready")

        # 检索相关内容
        logger.info(f"Streaming query for document {request.doc_id}: {request.query}")
        results = await rag_service.query(
            doc_id=request.doc_id,
            query_text=request.query,
            top_k=settings.retrieval_top_k
        )

        if not results:
            async def empty_response():
                import json
                # 先发送citations
                yield f"data: {json.dumps({'type': 'citations', 'data': []})}\n\n"
                # 再发送回答
                yield f"data: {json.dumps({'type': 'content', 'data': '抱歉，我在文档中没有找到相关信息。'})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                empty_response(),
                media_type="text/event-stream"
            )

        # 转换为引用格式
        citations = rag_service.format_citations(results)

        # 流式生成
        async def generate_stream():
            import json
            # 首先发送citations
            citations_data = [
                {
                    'id': c.id,
                    'source': c.source,
                    'page': c.page,
                    'snippet': c.snippet,
                    'content': c.content,
                    'type': c.type,
                    'block_id': c.block_id,
                    'bbox': c.bbox,
                    'image_path': c.image_path,
                    'score': c.score
                }
                for c in citations
            ]
            yield f"data: {json.dumps({'type': 'citations', 'data': citations_data})}\n\n"

            # 然后流式发送内容
            async for chunk in llm_service.generate_answer_stream(
                query=request.query,
                context_chunks=results,
                citations=citations
            ):
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"

            # 发送完成标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除文档及其索引

    Args:
        doc_id: 文档ID

    Returns:
        删除结果
    """
    try:
        # 删除索引
        await rag_service.delete_document(doc_id)

        # 删除进度记录
        if doc_id in indexing_progress:
            del indexing_progress[doc_id]

        # 删除文件
        upload_dir = Path(settings.upload_dir) / doc_id
        if upload_dir.exists():
            import shutil
            shutil.rmtree(upload_dir)

        logger.info(f"Document {doc_id} deleted")

        return {"message": "Document deleted successfully", "doc_id": doc_id}

    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
