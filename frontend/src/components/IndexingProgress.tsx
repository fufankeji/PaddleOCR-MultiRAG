import { motion } from 'framer-motion';
import { Upload, FileSearch, Cpu, CheckCircle2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getIndexingProgress } from '@/lib/api';
import { useDocument } from '@/contexts/DocumentContext';
import { Progress } from './ui/progress';

interface IndexingProgressProps {
  onComplete: () => void;
}

export function IndexingProgress({ onComplete }: IndexingProgressProps) {
  const { docId, setStats } = useDocument();
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('开始处理文档...');
  const [status, setStatus] = useState<'processing' | 'completed' | 'failed'>('processing');

  useEffect(() => {
    if (!docId) return;

    // 轮询进度
    const pollInterval = setInterval(async () => {
      try {
        const progressData = await getIndexingProgress(docId);

        setProgress(progressData.progress);
        setMessage(progressData.message);
        setStatus(progressData.status);

        if (progressData.status === 'completed') {
          clearInterval(pollInterval);
          if (progressData.stats) {
            setStats(progressData.stats);
          }
          setTimeout(() => {
            onComplete();
          }, 1000);
        } else if (progressData.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('获取进度失败:', error);
      }
    }, 1000); // 每秒轮询一次

    return () => clearInterval(pollInterval);
  }, [docId, onComplete, setStats]);

  // 根据进度判断当前步骤
  const getCurrentStep = () => {
    if (progress < 25) return 0; // 上传
    if (progress < 50) return 1; // 解析
    if (progress < 90) return 2; // 嵌入
    return 3; // 完成
  };

  const currentStep = getCurrentStep();

  const steps = [
    { icon: Upload, label: '上传' },
    { icon: FileSearch, label: 'OCR解析' },
    { icon: Cpu, label: '向量化' },
    { icon: CheckCircle2, label: '完成' },
  ];

  return (
    <div className="bg-card rounded-3xl p-8 border border-border shadow-lg space-y-6">
      <div className="text-center">
        <h3 className="text-accent mb-2">索引构建中</h3>
        <p className="text-sm text-muted-foreground">{message}</p>
      </div>

      {/* 进度条 */}
      <div className="space-y-2">
        <Progress value={progress} className="w-full" />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{progress}%</span>
          <span>{status === 'processing' ? '处理中...' : status === 'completed' ? '完成' : '失败'}</span>
        </div>
      </div>

      {/* 步骤指示器 */}
      <div className="flex items-center justify-between max-w-lg mx-auto pt-4">
        {steps.map((step, index) => {
          const Icon = step.icon;
          const isActive = index === currentStep;
          const isCompleted = index < currentStep;

          return (
            <div key={step.label} className="flex items-center">
              <div className="flex flex-col items-center gap-2">
                <motion.div
                  className={`rounded-full p-4 transition-all duration-500 ${
                    isActive
                      ? 'bg-gradient-to-r from-primary to-purple-600 shadow-[0_0_30px_rgba(74,0,224,0.6)]'
                      : isCompleted
                      ? 'bg-accent/20 border-2 border-accent'
                      : 'bg-muted/30 border-2 border-border'
                  }`}
                  animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                >
                  <Icon
                    className={`h-6 w-6 ${
                      isActive || isCompleted ? 'text-accent' : 'text-muted-foreground'
                    }`}
                  />
                </motion.div>
                <span
                  className={`text-xs ${
                    isActive || isCompleted ? 'text-accent' : 'text-muted-foreground'
                  }`}
                >
                  {step.label}
                </span>
              </div>

              {index < steps.length - 1 && (
                <div className="w-16 h-0.5 mx-2 mb-6">
                  <div
                    className={`h-full transition-all duration-500 ${
                      isCompleted ? 'bg-accent' : 'bg-border'
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
