import { useMemo, useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface ContentRendererProps {
  content: string;
  type: 'text' | 'table' | 'formula' | 'image';
  label: string;
}

export function ContentRenderer({ content, type, label }: ContentRendererProps) {
  const formulaRef = useRef<HTMLDivElement>(null);

  // HTML表格渲染
  const renderTable = (htmlContent: string) => {
    // 确保HTML表格有完整的结构
    let tableHtml = htmlContent;

    // 如果没有table标签,添加上
    if (!tableHtml.includes('<table')) {
      tableHtml = `<table>${tableHtml}</table>`;
    }

    // 添加表格样式
    const styledHtml = tableHtml
      .replace(/<table>/gi, '<table class="w-full border-collapse border border-border">')
      .replace(/<tr>/gi, '<tr class="border-b border-border">')
      .replace(/<td>/gi, '<td class="border border-border px-3 py-2">')
      .replace(/<td\s+colspan="(\d+)">/gi, '<td colspan="$1" class="border border-border px-3 py-2 text-center font-semibold">')
      .replace(/<td\s+rowspan="(\d+)">/gi, '<td rowspan="$1" class="border border-border px-3 py-2">');

    return (
      <div className="overflow-x-auto">
        <div
          className="text-sm"
          dangerouslySetInnerHTML={{ __html: styledHtml }}
        />
      </div>
    );
  };

  // LaTeX公式渲染 - 使用KaTeX
  useEffect(() => {
    if ((type === 'formula' || label.toLowerCase().includes('formula')) && formulaRef.current && content) {
      try {
        // 清理LaTeX字符串
        let latexStr = content.trim();

        // 移除$$包裹符号
        if (latexStr.startsWith('$$') && latexStr.endsWith('$$')) {
          latexStr = latexStr.slice(2, -2);
        } else if (latexStr.startsWith('$') && latexStr.endsWith('$')) {
          latexStr = latexStr.slice(1, -1);
        }

        // 使用KaTeX渲染
        katex.render(latexStr, formulaRef.current, {
          throwOnError: false,
          displayMode: true,
          output: 'html',
          trust: true,
        });
      } catch (error) {
        console.error('KaTeX rendering error:', error);
        // 如果渲染失败,显示原始LaTeX
        if (formulaRef.current) {
          formulaRef.current.textContent = content;
        }
      }
    }
  }, [content, type, label]);

  // Markdown文本渲染
  const renderMarkdown = (text: string) => {
    // 处理加粗
    let rendered = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // 处理斜体
    rendered = rendered.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // 处理代码
    rendered = rendered.replace(/`(.*?)`/g, '<code class="bg-muted px-1 rounded text-accent">$1</code>');
    // 处理换行
    rendered = rendered.replace(/\n/g, '<br/>');

    return (
      <div
        className="text-sm leading-relaxed"
        dangerouslySetInnerHTML={{ __html: rendered }}
      />
    );
  };

  // 根据类型返回不同的渲染结果
  if (!content) {
    return <span className="text-muted-foreground">(无内容)</span>;
  }

  if (type === 'formula' || label.toLowerCase().includes('formula')) {
    return (
      <div className="bg-card/50 rounded-lg p-4 my-2 overflow-x-auto">
        <div ref={formulaRef} className="text-center" />
      </div>
    );
  } else if (type === 'table' || label.toLowerCase().includes('table')) {
    return <div className="text-foreground">{renderTable(content)}</div>;
  } else {
    return <div className="text-foreground">{renderMarkdown(content)}</div>;
  }
}
