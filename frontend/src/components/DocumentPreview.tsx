import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { FileText, Scan, Image as ImageIcon, X } from 'lucide-react';
import { useDocument } from '@/contexts/DocumentContext';
import { useState, useEffect } from 'react';

interface Visualization {
  original: { name: string; path: string; type: string }[];
  layout_det: { name: string; path: string }[];
  layout_order: { name: string; path: string }[];
}

export function DocumentPreview() {
  const { fileName, stats, docId } = useDocument();
  const [visualizations, setVisualizations] = useState<Visualization | null>(null);
  const [loading, setLoading] = useState(false);
  const [enlargedImage, setEnlargedImage] = useState<string | null>(null);

  useEffect(() => {
    if (docId) {
      loadVisualizations();
    }
  }, [docId]);

  const loadVisualizations = async () => {
    if (!docId) return;

    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8100/api/documents/${docId}/visualizations`);
      if (response.ok) {
        const data = await response.json();
        setVisualizations(data);
      }
    } catch (error) {
      console.error('Failed to load visualizations:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-card rounded-3xl p-4 border border-border shadow-lg">
      <Tabs defaultValue="info" className="w-full">
        <TabsList className="grid w-full grid-cols-3 bg-muted/50">
          <TabsTrigger
            value="info"
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-purple-600 data-[state=active]:text-primary-foreground"
          >
            <FileText className="h-4 w-4 mr-2" />
            文档信息
          </TabsTrigger>
          <TabsTrigger
            value="structure"
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-purple-600 data-[state=active]:text-primary-foreground"
          >
            <Scan className="h-4 w-4 mr-2" />
            文档结构
          </TabsTrigger>
          <TabsTrigger
            value="visualization"
            className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-purple-600 data-[state=active]:text-primary-foreground"
          >
            <ImageIcon className="h-4 w-4 mr-2" />
            可视化对比
          </TabsTrigger>
        </TabsList>

        <TabsContent value="info" className="mt-3">
          <div className="bg-muted/20 rounded-2xl p-4 border border-border/50">
            <div className="space-y-3">
              <div className="flex items-center justify-center mb-4">
                <div className="bg-card/50 rounded-xl p-4 inline-block">
                  <FileText className="h-12 w-12 text-accent mx-auto" />
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between px-3 py-2 bg-card/50 rounded-lg">
                  <span className="text-xs text-muted-foreground">文件名</span>
                  <span className="text-xs text-foreground font-medium truncate max-w-[60%]">{fileName || '未知'}</span>
                </div>

                <div className="flex items-center justify-between px-3 py-2 bg-card/50 rounded-lg">
                  <span className="text-xs text-muted-foreground">处理状态</span>
                  <span className="text-xs text-accent font-medium">已完成</span>
                </div>

                <div className="flex items-center justify-between px-3 py-2 bg-card/50 rounded-lg">
                  <span className="text-xs text-muted-foreground">总内容块</span>
                  <span className="text-xs text-foreground font-medium">{stats?.total_blocks || 0}</span>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="structure" className="mt-3">
          <div className="bg-muted/20 rounded-2xl p-4 border border-border/50">
            {/* Simulated parsed view with colored bounding boxes */}
            <div className="space-y-2">
              {stats && stats.text_blocks > 0 && (
                <div className="border-2 border-blue-500/50 bg-blue-500/10 rounded-lg p-2">
                  <div className="h-3 bg-blue-500/30 rounded w-3/4 mb-1" />
                  <div className="h-3 bg-blue-500/30 rounded w-full mb-1" />
                  <div className="h-3 bg-blue-500/30 rounded w-5/6" />
                  <span className="text-xs text-blue-400 mt-1 inline-block">
                    文本块 ({stats.text_blocks})
                  </span>
                </div>
              )}

              {stats && stats.table_blocks > 0 && (
                <div className="border-2 border-green-500/50 bg-green-500/10 rounded-lg p-2">
                  <div className="grid grid-cols-3 gap-1">
                    <div className="h-2 bg-green-500/30 rounded" />
                    <div className="h-2 bg-green-500/30 rounded" />
                    <div className="h-2 bg-green-500/30 rounded" />
                    <div className="h-2 bg-green-500/30 rounded" />
                    <div className="h-2 bg-green-500/30 rounded" />
                    <div className="h-2 bg-green-500/30 rounded" />
                  </div>
                  <span className="text-xs text-green-400 mt-1 inline-block">
                    表格 ({stats.table_blocks})
                  </span>
                </div>
              )}

              {stats && stats.image_blocks > 0 && (
                <div className="border-2 border-purple-500/50 bg-purple-500/10 rounded-lg p-2">
                  <div className="h-16 bg-purple-500/30 rounded" />
                  <span className="text-xs text-purple-400 mt-1 inline-block">
                    图像 ({stats.image_blocks})
                  </span>
                </div>
              )}

              {stats && stats.formula_blocks > 0 && (
                <div className="border-2 border-pink-500/50 bg-pink-500/10 rounded-lg p-2">
                  <div className="h-6 bg-pink-500/30 rounded" />
                  <span className="text-xs text-pink-400 mt-1 inline-block">
                    公式 ({stats.formula_blocks})
                  </span>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="visualization" className="mt-3">
          <div className="bg-muted/20 rounded-2xl p-4 border border-border/50">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-muted-foreground">加载中...</div>
              </div>
            ) : visualizations ? (
              <div className="space-y-6">
                {/* 原始文件 */}
                {visualizations.original.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-3">原始文件</h4>
                    <div className="grid grid-cols-1 gap-4">
                      {visualizations.original.map((file) => (
                        <div key={file.path} className="bg-card/50 rounded-lg p-4 border border-border/50">
                          <div className="text-xs text-muted-foreground mb-2">{file.name}</div>
                          {file.type.startsWith('image/') ? (
                            <img
                              src={`http://localhost:8100${file.path}`}
                              alt={file.name}
                              className="max-w-full h-auto rounded border border-border cursor-pointer hover:opacity-80 transition-opacity"
                              onClick={() => setEnlargedImage(`http://localhost:8100${file.path}`)}
                            />
                          ) : (
                            <div className="text-sm text-foreground">
                              PDF 文件 - <a href={`http://localhost:8100${file.path}`} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">查看</a>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 布局检测结果 */}
                {visualizations.layout_det.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-3">布局检测结果</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {visualizations.layout_det.map((file) => (
                        <div key={file.path} className="bg-card/50 rounded-lg p-4 border border-border/50">
                          <div className="text-xs text-muted-foreground mb-2">{file.name}</div>
                          <img
                            src={`http://localhost:8100${file.path}`}
                            alt={file.name}
                            className="max-w-full h-auto rounded border border-border cursor-pointer hover:opacity-80 transition-opacity"
                            onClick={() => setEnlargedImage(`http://localhost:8100${file.path}`)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 阅读顺序可视化 */}
                {visualizations.layout_order.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-muted-foreground mb-3">阅读顺序可视化</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {visualizations.layout_order.map((file) => (
                        <div key={file.path} className="bg-card/50 rounded-lg p-4 border border-border/50">
                          <div className="text-xs text-muted-foreground mb-2">{file.name}</div>
                          <img
                            src={`http://localhost:8100${file.path}`}
                            alt={file.name}
                            className="max-w-full h-auto rounded border border-border cursor-pointer hover:opacity-80 transition-opacity"
                            onClick={() => setEnlargedImage(`http://localhost:8100${file.path}`)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {visualizations.original.length === 0 &&
                 visualizations.layout_det.length === 0 &&
                 visualizations.layout_order.length === 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    暂无可视化数据
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                暂无可视化数据
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>

      {/* 图片放大模态框 */}
      {enlargedImage && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setEnlargedImage(null)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]">
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute -top-12 right-0 text-white hover:text-gray-300 transition-colors"
            >
              <X className="w-8 h-8" />
            </button>
            <img
              src={enlargedImage}
              alt="放大查看"
              className="max-w-full max-h-[90vh] object-contain rounded"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  );
}
