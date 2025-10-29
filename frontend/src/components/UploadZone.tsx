import { Upload, FileText } from 'lucide-react';
import { useState } from 'react';
import { Progress } from './ui/progress';
import { uploadDocument } from '@/lib/api';
import { useDocument } from '@/contexts/DocumentContext';
import { useToast } from '@/hooks/use-toast';

interface UploadZoneProps {
  onUploadComplete: () => void;
}

export function UploadZone({ onUploadComplete }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const { setDocument } = useDocument();
  const { toast } = useToast();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleUpload(e.target.files[0]);
    }
  };

  const handleUpload = async (file: File) => {
    // 验证文件类型
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      toast({
        title: '文件类型不支持',
        description: '请上传 PDF、PNG 或 JPG 格式的文件',
        variant: 'destructive',
      });
      return;
    }

    // 验证文件大小 (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      toast({
        title: '文件过大',
        description: '文件大小不能超过 50MB',
        variant: 'destructive',
      });
      return;
    }

    setIsUploading(true);
    setProgress(0);

    // 模拟进度
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const response = await uploadDocument(file);

      clearInterval(progressInterval);
      setProgress(100);

      // 保存文档信息到 context
      setDocument(response.doc_id, response.file_name);

      toast({
        title: '上传成功',
        description: `文件 ${response.file_name} 已开始处理`,
      });

      setTimeout(() => {
        onUploadComplete();
      }, 500);
    } catch (error) {
      clearInterval(progressInterval);
      setIsUploading(false);
      setProgress(0);

      toast({
        title: '上传失败',
        description: error instanceof Error ? error.message : '请稍后重试',
        variant: 'destructive',
      });
    }
  };

  return (
    <div
      className={`relative border-2 border-dashed rounded-3xl p-12 text-center transition-all duration-300 ${
        isDragging
          ? 'border-accent bg-accent/10 shadow-[0_0_20px_rgba(0,217,255,0.3)]'
          : 'border-border bg-card/30 hover:border-accent/50'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isUploading ? (
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className="relative">
              <FileText className="h-16 w-16 text-accent animate-pulse" />
              <div className="absolute inset-0 bg-accent/20 rounded-full blur-xl animate-pulse" />
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-muted-foreground">上传中...</p>
            <Progress value={progress} className="w-full" />
            <p className="text-xs text-muted-foreground">{progress}%</p>
          </div>
        </div>
      ) : (
        <>
          <div className="flex justify-center mb-4">
            <div className="relative">
              <Upload className="h-16 w-16 text-accent" />
              <div className="absolute inset-0 bg-accent/20 rounded-full blur-xl" />
            </div>
          </div>
          <h3 className="mb-2 text-foreground">拖放文件到这里</h3>
          <p className="text-sm text-muted-foreground mb-4">
            支持 PDF、PNG、JPG 文件（最大 50MB）
          </p>
          <label className="inline-block">
            <input
              type="file"
              className="hidden"
              accept=".pdf,.png,.jpg,.jpeg"
              onChange={handleFileSelect}
            />
            <span className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary to-purple-600 text-primary-foreground rounded-xl cursor-pointer hover:shadow-[0_0_20px_rgba(74,0,224,0.5)] transition-all duration-300">
              <Upload className="h-4 w-4" />
              选择文件
            </span>
          </label>
        </>
      )}
    </div>
  );
}
