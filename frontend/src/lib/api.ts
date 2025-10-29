/**
 * API Client for Backend Communication
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface UploadResponse {
  doc_id: string;
  file_name: string;
  status: string;
  message: string;
}

export interface DocumentStats {
  doc_id: string;
  text_blocks: number;
  table_blocks: number;
  image_blocks: number;
  formula_blocks: number;
  total_blocks: number;
}

export interface IndexingProgress {
  doc_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  stats?: DocumentStats;
}

export interface Citation {
  id: number;
  source: string;
  page: number;
  snippet: string;
  type: string;
  block_id?: number;
  bbox?: number[];
  score: number;
}

export interface QueryRequest {
  doc_id: string;
  query: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  doc_id: string;
}

/**
 * Upload a document
 */
export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Upload failed');
  }

  return response.json();
}

/**
 * Get indexing progress
 */
export async function getIndexingProgress(docId: string): Promise<IndexingProgress> {
  const response = await fetch(`${API_BASE_URL}/api/progress/${docId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get progress');
  }

  return response.json();
}

/**
 * Query document
 */
export async function queryDocument(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Query failed');
  }

  return response.json();
}

/**
 * Query document with streaming
 */
export async function* queryDocumentStream(
  request: QueryRequest
): AsyncGenerator<string, void, unknown> {
  const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Query failed');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data) {
          yield data;
        }
      }
    }
  }
}

/**
 * Delete document
 */
export async function deleteDocument(docId: string): Promise<{ message: string; doc_id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/documents/${docId}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Delete failed');
  }

  return response.json();
}

/**
 * Get document blocks
 */
export interface DocumentBlock {
  block_id: number;
  block_label: string;
  block_content: string;
  block_bbox: number[];
  block_order: number | null;
  page_index?: number; // 页码
  source_file?: string; // 源文件名
  image_path?: string; // 图片路径(仅image类型的block有)
}

export async function getDocumentBlocks(
  docId: string,
  blockType?: 'text' | 'table' | 'image' | 'formula',
  page?: number
): Promise<{ blocks: DocumentBlock[]; total: number }> {
  const params = new URLSearchParams();
  if (blockType) params.append('block_type', blockType);
  if (page !== undefined) params.append('page', page.toString());

  const url = `${API_BASE_URL}/api/documents/${docId}/blocks${params.toString() ? '?' + params.toString() : ''}`;

  const response = await fetch(url);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get blocks');
  }

  return response.json();
}

/**
 * Health check
 */
export async function healthCheck(): Promise<{ status: string; message: string; version: string }> {
  const response = await fetch(`${API_BASE_URL}/`);

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}
