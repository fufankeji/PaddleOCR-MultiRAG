import { useState } from 'react';
import { ChevronDown, ChevronUp, Hash, MapPin } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

interface Citation {
  id: number;
  source: string;
  page: number;
  snippet: string;
  content?: string;
  type: 'text' | 'table' | 'image' | 'formula';
  block_id?: number;
  bbox?: number[];
  image_path?: string;
  score?: number;
}

interface Message {
  content: string;
  isUser: boolean;
  citations?: Citation[];
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const { content, isUser, citations = [] } = message;
  const [expandedCitation, setExpandedCitation] = useState<number | null>(null);

  // 渲染内容，支持 Markdown 并处理引用标记
  const renderContent = () => {
    // 替换 【数字】 为占位符，稍后用组件替换
    const processedContent = content.replace(/【(\d+)】/g, (match, id) => {
      return `<cite data-id="${id}">[${id}]</cite>`;
    });

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          // 自定义渲染引用标记
          cite: ({ node, ...props }: any) => {
            const citationId = parseInt(props['data-id'] || '0');
            return (
              <sup
                onClick={() => setExpandedCitation(expandedCitation === citationId ? null : citationId)}
                className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 ml-0.5 text-xs bg-cyan-500 text-white rounded-full cursor-pointer hover:bg-cyan-600 transition-colors"
              >
                {citationId}
              </sup>
            );
          },
          // 段落样式
          p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
          // 列表样式
          ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
          ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
          li: ({ node, ...props }) => <li className="ml-2" {...props} />,
          // 代码块样式
          code: ({ node, inline, ...props }: any) =>
            inline ? (
              <code className="bg-slate-700 px-1.5 py-0.5 rounded text-cyan-300 text-xs" {...props} />
            ) : (
              <code className="block bg-slate-900 p-3 rounded-lg my-2 text-sm overflow-x-auto" {...props} />
            ),
          // 强调样式
          strong: ({ node, ...props }) => <strong className="font-bold text-cyan-300" {...props} />,
          em: ({ node, ...props }) => <em className="italic text-slate-300" {...props} />,
          // 链接样式
          a: ({ node, ...props }) => (
            <a className="text-cyan-400 hover:text-cyan-300 underline" target="_blank" rel="noopener noreferrer" {...props} />
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    );
  };

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[70%] bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-lg">
          <div className="text-sm leading-relaxed whitespace-pre-wrap">{content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6">
      {/* 助手消息气泡 */}
      <div className="flex justify-start mb-2">
        <div className="max-w-[85%] bg-slate-800/90 border border-cyan-500/30 text-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-lg">
          <div className="text-sm leading-relaxed">
            {renderContent()}
          </div>
        </div>
      </div>

      {/* 展开的引用卡片 */}
      {expandedCitation !== null && citations.find(c => c.id === expandedCitation) && (
        <div className="ml-4 mt-2 bg-slate-800/50 border border-cyan-500/30 rounded-xl p-4 max-w-[80%] animate-in fade-in slide-in-from-top-2 duration-200">
          {(() => {
            const citation = citations.find(c => c.id === expandedCitation)!;
            return (
              <>
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 space-y-2">
                    {/* 来源信息 */}
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-cyan-400 font-medium">
                        {citation.source} - 第 {citation.page} 页
                      </span>
                      <span className="text-slate-400">
                        类型: {citation.type === 'text' ? '文本' : citation.type === 'table' ? '表格' : citation.type === 'formula' ? '公式' : '图像'}
                      </span>
                    </div>

                    {/* Block ID 和相关性 */}
                    <div className="flex items-center gap-3 text-xs text-slate-400">
                      {citation.block_id !== undefined && (
                        <span className="flex items-center gap-1">
                          <Hash className="h-3 w-3" />
                          Block {citation.block_id}
                        </span>
                      )}
                      {citation.score !== undefined && (
                        <span className="text-cyan-400 font-medium">
                          相关性: {(citation.score * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>

                    {/* 位置信息 */}
                    {citation.bbox && citation.bbox.length === 4 && (
                      <div className="flex items-center gap-1 text-xs text-slate-500">
                        <MapPin className="h-3 w-3" />
                        位置: [{citation.bbox.join(', ')}]
                      </div>
                    )}
                  </div>

                  {/* 关闭按钮 */}
                  <button
                    onClick={() => setExpandedCitation(null)}
                    className="text-slate-400 hover:text-cyan-400 transition-colors ml-2 flex-shrink-0"
                  >
                    <ChevronUp className="h-4 w-4" />
                  </button>
                </div>

                {/* 引用内容 */}
                <div className="bg-slate-900/50 rounded-lg p-3 text-sm text-slate-300 leading-relaxed border border-slate-700/50">
                  {/* 如果有图片路径，显示图片 */}
                  {citation.image_path && (
                    <div className="mb-3">
                      <img
                        src={`http://localhost:8100${citation.image_path}`}
                        alt={`引用 ${citation.id}`}
                        className="max-w-full h-auto rounded border border-slate-600"
                      />
                    </div>
                  )}

                  {/* 显示内容 */}
                  {citation.type === 'table' ? (
                    <div
                      className="overflow-x-auto"
                      dangerouslySetInnerHTML={{ __html: citation.content || citation.snippet }}
                    />
                  ) : (
                    <div className="whitespace-pre-wrap">
                      {citation.content || citation.snippet}
                    </div>
                  )}
                </div>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}
