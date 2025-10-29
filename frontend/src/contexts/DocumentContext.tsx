import React, { createContext, useContext, useState, ReactNode } from 'react';
import { DocumentStats } from '@/lib/api';

interface DocumentContextType {
  docId: string | null;
  fileName: string | null;
  stats: DocumentStats | null;
  setDocument: (docId: string, fileName: string, stats?: DocumentStats) => void;
  setStats: (stats: DocumentStats) => void;
  clearDocument: () => void;
}

const DocumentContext = createContext<DocumentContextType | undefined>(undefined);

export function DocumentProvider({ children }: { children: ReactNode }) {
  const [docId, setDocId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [stats, setStatsState] = useState<DocumentStats | null>(null);

  const setDocument = (newDocId: string, newFileName: string, newStats?: DocumentStats) => {
    setDocId(newDocId);
    setFileName(newFileName);
    if (newStats) {
      setStatsState(newStats);
    }
  };

  const setStats = (newStats: DocumentStats) => {
    setStatsState(newStats);
  };

  const clearDocument = () => {
    setDocId(null);
    setFileName(null);
    setStatsState(null);
  };

  return (
    <DocumentContext.Provider value={{ docId, fileName, stats, setDocument, setStats, clearDocument }}>
      {children}
    </DocumentContext.Provider>
  );
}

export function useDocument() {
  const context = useContext(DocumentContext);
  if (context === undefined) {
    throw new Error('useDocument must be used within a DocumentProvider');
  }
  return context;
}
