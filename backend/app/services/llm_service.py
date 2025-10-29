"""
LLM Service for Alibaba Cloud DashScope (Qwen)
"""

import logging
import json
from typing import List, Dict, Any, Optional
import asyncio

import dashscope
from dashscope import Generation

from app.config import settings
from app.models.schemas import Citation

logger = logging.getLogger(__name__)


class LLMService:
    """阿里云百炼平台 Qwen 模型服务"""

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """初始化 LLM 服务"""
        if self._initialized:
            return

        try:
            logger.info("Initializing LLM service...")

            # 设置 DashScope API Key
            dashscope.api_key = settings.dashscope_api_key

            self._initialized = True
            logger.info("LLM service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ) -> str:
        """
        基于检索上下文生成回答

        Args:
            query: 用户查询
            context_chunks: 检索到的上下文块
            citations: 引用列表

        Returns:
            生成的回答文本
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 构建上下文
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                metadata = chunk["metadata"]
                content = chunk["content"]
                block_type = metadata.get("block_type", "text")
                page = metadata.get("page_index", 0)

                # 根据块类型添加前缀
                if block_type == "table":
                    prefix = f"[表格 - 第{page}页]"
                elif block_type == "image":
                    prefix = f"[图像/图表 - 第{page}页]"
                elif block_type == "formula":
                    prefix = f"[公式 - 第{page}页]"
                else:
                    prefix = f"[文本 - 第{page}页]"

                context_parts.append(f"{prefix}\n{content}")

            context = "\n\n".join(context_parts)

            # 构建引用映射（用于在回答中标注引用）
            citation_map = {}
            for citation in citations:
                citation_map[citation.id] = f"【{citation.id}】"

            # 构建 prompt
            system_prompt = """你是一个专业的文档问答助手。你的任务是：
1. 基于提供的文档上下文，准确回答用户的问题
2. 在回答中使用【数字】标记引用来源（例如【1】【2】）
3. 对于表格、图像、公式等特殊内容，明确指出其类型
4. 如果上下文中没有相关信息，诚实地说明
5. 回答要准确、简洁、结构清晰

引用标注规则：
- 使用【1】【2】【3】等数字标记，对应检索到的文档块
- 每个关键信息点都应该标注引用来源
- 多个来源可以连续标注，如【1】【2】
"""

            user_prompt = f"""基于以下文档上下文，回答用户的问题。

## 文档上下文
{context}

## 用户问题
{query}

请提供详细的回答，并在关键信息处标注引用来源【1】【2】等。
"""

            # 调用 Qwen 模型
            logger.info(f"Calling Qwen model: {settings.qwen_model_name}")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._call_qwen_sync,
                system_prompt,
                user_prompt
            )

            # 处理响应 - Qwen使用message格式返回
            if hasattr(response, 'output'):
                # 尝试从choices中获取（新的API格式）
                if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                    choice = response.output.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        answer = choice.message.content
                    else:
                        answer = ''
                # 尝试从text字段获取（旧的API格式）
                elif hasattr(response.output, 'text'):
                    answer = response.output.text
                elif isinstance(response.output, dict):
                    answer = response.output.get('text', '')
                else:
                    answer = ''
            else:
                answer = ''

            # 如果answer为None,设置默认值
            if answer is None:
                answer = ''

            logger.info(f"Generated answer: {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _call_qwen_sync(self, system_prompt: str, user_prompt: str):
        """同步调用 Qwen 模型"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = Generation.call(
            model=settings.qwen_model_name,
            messages=messages,
            result_format='message',
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API error: {response.code} - {response.message}")

        logger.info(f"Qwen response status: {response.status_code}")
        logger.info(f"Response hasattr output: {hasattr(response, 'output')}")
        if hasattr(response, 'output'):
            logger.info(f"Response output type: {type(response.output)}")
            logger.info(f"Response output: {response.output}")

        return response

    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[Citation]
    ):
        """
        流式生成回答（支持 SSE）

        Args:
            query: 用户查询
            context_chunks: 检索到的上下文块
            citations: 引用列表

        Yields:
            生成的回答片段
        """
        if not self._initialized:
            await self.initialize()

        try:
            # 构建上下文（与普通生成相同）
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                metadata = chunk["metadata"]
                content = chunk["content"]
                block_type = metadata.get("block_type", "text")
                page = metadata.get("page_index", 0)

                if block_type == "table":
                    prefix = f"[表格 - 第{page}页]"
                elif block_type == "image":
                    prefix = f"[图像/图表 - 第{page}页]"
                elif block_type == "formula":
                    prefix = f"[公式 - 第{page}页]"
                else:
                    prefix = f"[文本 - 第{page}页]"

                context_parts.append(f"{prefix}\n{content}")

            context = "\n\n".join(context_parts)

            system_prompt = """你是一个专业的文档问答助手。你的任务是：
1. 基于提供的文档上下文，准确回答用户的问题
2. 在回答中使用【数字】标记引用来源（例如【1】【2】）
3. 对于表格、图像、公式等特殊内容，明确指出其类型
4. 如果上下文中没有相关信息，诚实地说明
5. 回答要准确、简洁、结构清晰

引用标注规则：
- 使用【1】【2】【3】等数字标记，对应检索到的文档块
- 每个关键信息点都应该标注引用来源
- 多个来源可以连续标注，如【1】【2】
"""

            user_prompt = f"""基于以下文档上下文，回答用户的问题。

## 文档上下文
{context}

## 用户问题
{query}

请提供详细的回答，并在关键信息处标注引用来源【1】【2】等。
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # 流式调用
            responses = Generation.call(
                model=settings.qwen_model_name,
                messages=messages,
                result_format='message',
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                stream=True,
                incremental_output=True
            )

            for response in responses:
                if response.status_code == 200:
                    # 处理流式响应 - 使用 message 格式
                    chunk = ''
                    if hasattr(response, 'output'):
                        # 尝试从choices中获取（message格式）
                        if hasattr(response.output, 'choices') and len(response.output.choices) > 0:
                            choice = response.output.choices[0]
                            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                                chunk = choice.message.content
                        # 尝试从text字段获取（旧格式）
                        elif hasattr(response.output, 'text'):
                            chunk = response.output.text
                        elif isinstance(response.output, dict):
                            chunk = response.output.get('text', '')

                    if chunk:
                        yield chunk
                else:
                    logger.error(f"Stream error: {response.code} - {response.message}")
                    break

        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            raise

    async def cleanup(self):
        """清理资源"""
        self._initialized = False
        logger.info("LLM service cleaned up")
