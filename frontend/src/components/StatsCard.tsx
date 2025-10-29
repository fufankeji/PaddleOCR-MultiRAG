import { FileText, Table2, Sigma, Image } from 'lucide-react';
import { useDocument } from '@/contexts/DocumentContext';
import { useState } from 'react';
import { getDocumentBlocks, DocumentBlock } from '@/lib/api';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { ContentRenderer } from './ContentRenderer';

export function StatsCard() {
  const { stats, docId } = useDocument();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [selectedType, setSelectedType] = useState<string>('');
  const [selectedBlockType, setSelectedBlockType] = useState<'text' | 'table' | 'image' | 'formula'>('text');
  const [blocks, setBlocks] = useState<DocumentBlock[]>([]);
  const [loading, setLoading] = useState(false);

  if (!stats) {
    return null;
  }

  const statsData = [
    {
      label: '文本',
      value: stats.text_blocks,
      icon: FileText,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
      type: 'text' as const
    },
    {
      label: '表格',
      value: stats.table_blocks,
      icon: Table2,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      type: 'table' as const
    },
    {
      label: '公式',
      value: stats.formula_blocks,
      icon: Sigma,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      type: 'formula' as const
    },
    {
      label: '图像',
      value: stats.image_blocks,
      icon: Image,
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/10',
      type: 'image' as const
    },
  ];

  const handleCardClick = async (stat: typeof statsData[0]) => {
    if (stat.value === 0 || !docId) return;

    setSelectedType(stat.label);
    setSelectedBlockType(stat.type);
    setIsDialogOpen(true);
    setLoading(true);

    try {
      const response = await getDocumentBlocks(docId, stat.type);
      setBlocks(response.blocks);
    } catch (error) {
      console.error('Failed to load blocks:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="bg-card rounded-3xl p-4 border border-border shadow-lg">
        <h3 className="mb-3 text-sm text-muted-foreground">提取摘要</h3>
        <div className="grid grid-cols-2 gap-2">
          {statsData.map((stat) => {
            const Icon = stat.icon;
            return (
              <div
                key={stat.label}
                onClick={() => handleCardClick(stat)}
                className={`${stat.bgColor} ${stat.color} rounded-xl p-3 text-left transition-all duration-300 hover:scale-105 hover:shadow-[0_0_20px_rgba(0,217,255,0.2)] border border-transparent hover:border-accent/30 ${
                  stat.value > 0 ? 'cursor-pointer' : 'opacity-50 cursor-not-allowed'
                }`}
              >
                <Icon className="h-4 w-4 mb-1" />
                <div className="text-xl mb-0.5">{stat.value}</div>
                <div className="text-xs opacity-80">{stat.label}</div>
              </div>
            );
          })}
        </div>
        <div className="mt-3 pt-3 border-t border-border/50">
          <div className="text-xs text-muted-foreground">
            <span className="text-accent font-medium">{stats.total_blocks}</span> 个内容块已索引
          </div>
        </div>
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] bg-background">
          <DialogHeader>
            <DialogTitle className="text-accent">
              {selectedType} - 提取内容
            </DialogTitle>
          </DialogHeader>
          <div className="overflow-y-auto max-h-[60vh] pr-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-muted-foreground">加载中...</div>
              </div>
            ) : (
              <div className="space-y-4">
                {blocks.map((block, index) => (
                  <div
                    key={block.block_id}
                    className="bg-muted/20 rounded-lg p-4 border border-border/50"
                  >
                    <div className="text-sm text-muted-foreground mb-2 flex items-center justify-between">
                      <span>#{index + 1} - {block.block_label}</span>
                      {block.page_index !== undefined && (
                        <span className="text-xs bg-accent/20 text-accent px-2 py-0.5 rounded">
                          第 {block.page_index + 1} 页
                        </span>
                      )}
                    </div>
                    {block.image_path ? (
                      <div className="mt-2">
                        <img
                          src={`http://localhost:8100${block.image_path}`}
                          alt={`Block ${block.block_id}`}
                          className="max-w-full h-auto rounded border border-border"
                        />
                      </div>
                    ) : (
                      <ContentRenderer
                        content={block.block_content}
                        type={selectedBlockType}
                        label={block.block_label}
                      />
                    )}
                  </div>
                ))}
                {blocks.length === 0 && !loading && (
                  <div className="text-center py-8 text-muted-foreground">
                    暂无内容
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
