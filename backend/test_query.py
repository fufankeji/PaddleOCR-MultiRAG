"""测试 ChromaDB 查询"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag_service import RAGService

async def main():
    doc_id = "fee551f0-e8ad-4959-9a8f-541ef6f74f48"
    query = "这个文档讲的是什么？"

    # 初始化服务
    rag_service = RAGService()
    await rag_service.initialize()

    print(f"查询文档: {doc_id}")
    print(f"查询问题: {query}")
    print("-" * 50)

    try:
        # 执行查询
        results = await rag_service.query(
            doc_id=doc_id,
            query_text=query,
            top_k=3
        )

        print(f"✅ 查询成功！找到 {len(results)} 个结果\n")

        for i, result in enumerate(results, 1):
            print(f"结果 {i}:")
            print(f"  内容: {result.get('content', '')[:100]}...")
            print(f"  类型: {result.get('metadata', {}).get('block_type', 'unknown')}")
            print(f"  距离: {result.get('distance', 0):.4f}")
            print()

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
