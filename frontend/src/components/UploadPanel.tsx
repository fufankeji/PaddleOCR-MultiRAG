import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadZone } from './UploadZone';
import { IndexingProgress } from './IndexingProgress';
import { DocumentPreview } from './DocumentPreview';
import { StatsCard } from './StatsCard';
import { CheckCircle2, RotateCcw } from 'lucide-react';
import { Button } from './ui/button';

type UploadState = 'idle' | 'indexing' | 'complete';

export function UploadPanel() {
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [showSuccessToast, setShowSuccessToast] = useState(false);

  const handleUploadComplete = () => {
    setUploadState('indexing');
  };

  const handleIndexingComplete = () => {
    setUploadState('complete');
    setShowSuccessToast(true);
    setTimeout(() => setShowSuccessToast(false), 3000);
  };

  const handleReset = () => {
    setUploadState('idle');
    setShowSuccessToast(false);
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-foreground">文档上传与索引</h2>
        {uploadState === 'complete' && (
          <Button
            onClick={handleReset}
            variant="outline"
            size="sm"
            className="border-accent/30 text-accent hover:bg-accent/10"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            重新上传
          </Button>
        )}
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto pr-2">
        {uploadState === 'idle' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <UploadZone onUploadComplete={handleUploadComplete} />
          </motion.div>
        )}

        {uploadState === 'indexing' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <IndexingProgress onComplete={handleIndexingComplete} />
          </motion.div>
        )}

        {uploadState === 'complete' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <DocumentPreview />
            <StatsCard />
          </motion.div>
        )}
      </div>

      <AnimatePresence>
        {showSuccessToast && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50"
          >
            <div className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-4 rounded-2xl shadow-[0_0_30px_rgba(34,197,94,0.5)] flex items-center gap-3">
              <CheckCircle2 className="h-6 w-6" />
              <div>
                <p className="font-medium">索引构建成功 ✅</p>
                <p className="text-sm opacity-90">准备就绪，可以开始问答</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
