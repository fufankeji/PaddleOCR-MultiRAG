import { useState, useRef, useEffect } from 'react';
import { flushSync } from 'react-dom';
import { Send, Sparkles } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { queryDocument } from '@/lib/api';
import { useDocument } from '@/contexts/DocumentContext';
import { useToast } from '@/hooks/use-toast';

interface Message {
  content: string;
  isUser: boolean;
  citations?: any[];
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { docId } = useDocument();
  const scrollRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [messages]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();

    if (!input.trim() || !docId || isLoading) return;

    const userMessage: Message = {
      content: input.trim(),
      isUser: true,
    };

    setMessages(prev => [...prev, userMessage]);
    const query = input.trim();
    setInput('');
    setIsLoading(true);

    // 创建一个空的助手消息用于流式更新
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [
      ...prev,
      {
        content: '正在思考...',
        isUser: false,
        citations: [],
      },
    ]);

    try {
      const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8100';
      const response = await fetch(`${apiUrl}/api/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          doc_id: docId,
        }),
      });

      if (!response.ok) {
        throw new Error('Stream query failed');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let citations: any[] = [];
      let isDone = false;

      if (!reader) {
        throw new Error('Response body is null');
      }

      while (!isDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              isDone = true;
              break;
            }

            try {
              const parsed = JSON.parse(data);
              if (parsed.type === 'citations') {
                citations = parsed.data;
                console.log('Received citations:', citations.length);
                // 更新citations，并清空"正在思考..."
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[assistantMessageIndex] = {
                    content: '',
                    isUser: false,
                    citations: citations,
                  };
                  return newMessages;
                });
              } else if (parsed.type === 'content') {
                // 流式更新内容
                console.log('Received content chunk:', parsed.data);
                // 使用 flushSync 强制立即更新，避免批处理
                flushSync(() => {
                  setMessages(prev => {
                    const newMessages = [...prev];
                    const currentContent = newMessages[assistantMessageIndex].content;
                    // 第一次收到内容时，如果还是"正在思考..."，先清空
                    const baseContent = currentContent === '正在思考...' ? '' : currentContent;
                    newMessages[assistantMessageIndex] = {
                      ...newMessages[assistantMessageIndex],
                      content: baseContent + parsed.data,
                    };
                    return newMessages;
                  });
                });
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }

      console.log('Stream processing completed successfully');
    } catch (error) {
      toast({
        title: '查询失败',
        description: error instanceof Error ? error.message : '请稍后重试',
        variant: 'destructive',
      });

      // 更新为错误消息
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[assistantMessageIndex] = {
          content: '抱歉，查询时出现了错误。请稍后再试。',
          isUser: false,
        };
        return newMessages;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <h2 className="text-foreground mb-4">交互式问答</h2>

      <div className="flex-1 flex flex-col bg-card/30 backdrop-blur-sm rounded-3xl border border-border shadow-lg overflow-hidden">
        {/* Messages Area */}
        <ScrollArea className="flex-1 overflow-y-auto p-6">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 mx-auto rounded-full bg-accent/10 flex items-center justify-center">
                  <Sparkles className="h-8 w-8 text-accent" />
                </div>
                <h3 className="text-accent">
                  {docId ? '准备就绪' : '等待文档上传'}
                </h3>
                <p className="text-sm text-muted-foreground max-w-md">
                  {docId
                    ? '文档已索引完成，您可以开始提问了'
                    : '请先上传文档，完成索引后即可开始智能问答'}
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))}
              <div ref={scrollRef} />
            </div>
          )}
        </ScrollArea>

        {/* Input Area - Fixed at bottom */}
        <div className="flex-shrink-0 p-6 border-t border-border/50 bg-card/50">
          <form onSubmit={handleSubmit} className="flex items-start gap-3">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!docId || isLoading}
              placeholder={
                !docId
                  ? '请先上传文档...'
                  : isLoading
                  ? '正在生成回答...'
                  : '询问任何关于您文档的问题...'
              }
              className="flex-1 resize-none bg-popover text-foreground placeholder:text-muted-foreground disabled:opacity-50 min-h-[42px]"
              rows={1}
            />
            <Button
              type="submit"
              disabled={!docId || !input.trim() || isLoading}
              className="flex-shrink-0 bg-gradient-to-r from-primary to-purple-600 hover:shadow-[0_0_20px_rgba(74,0,224,0.5)] transition-all duration-300 h-[42px] px-4 flex items-center justify-center"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
