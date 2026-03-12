'use client';
import { useState, useRef, useEffect } from 'react';
import { pollStatus, getDocuments, deleteDocument } from '../../lib/api';
import axios from 'axios';

interface UploadedDoc {
  filename: string;
  doc_type: string;
  chunks?: number;
  pdf_id: string;
}

interface Props {
  sessionId: string;
  onUploadComplete: (filename: string) => void;
}

export default function UploadPanel({ sessionId, onUploadComplete }: Props) {
  const [uploads, setUploads] = useState<UploadedDoc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState<string>('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const docs = await getDocuments(sessionId);
        setUploads(docs);
      } catch (e) {
        console.error('Failed to fetch documents', e);
      }
    };
    setUploads([]);
    fetchDocs();
  }, [sessionId]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);

    for (const file of Array.from(files)) {
      if (!file.name.endsWith('.pdf')) {
        setStatus(`✗ ${file.name} is not a PDF`);
        continue;
      }

      setStatus(`Uploading ${file.name}...`);

      try {
        const form = new FormData();
        form.append('file', file);
        form.append('session_id', sessionId);

        const { data } = await axios.post(
          'http://localhost:8000/upload',
          form,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        );

        const jobId = data.job_id;
        setStatus(`Processing ${file.name}...`);

        const poll = async () => {
          const result = await pollStatus(jobId);
          if (result.status === 'success') {
            const doc: UploadedDoc = {
              filename: file.name,
              doc_type: result.doc_type || 'general',
              chunks: result.chunk_count,
              pdf_id: data.doc_id
            };
            setUploads(prev => [...prev, doc]);
            setStatus('');
            onUploadComplete(file.name);
          } else if (result.status === 'failed') {
            setStatus(`✗ ${file.name} failed`);
          } else {
            setStatus(`${file.name}: ${result.status}...`);
            setTimeout(poll, 2000);
          }
        };
        await poll();

      } catch (err: any) {
        const detail = err?.response?.data?.detail || `Upload failed for ${file.name}`;
        setStatus(`✗ ${detail}`);
      }
    }

    setUploading(false);
    if (fileRef.current) fileRef.current.value = '';
  };

  const handleDeleteConfirm = async (doc: UploadedDoc) => {
    if (!doc.pdf_id) return;
    setConfirmingId(null);
    setDeletingId(doc.pdf_id);
    try {
      await deleteDocument(sessionId, doc.pdf_id);
      setUploads(prev => prev.filter(d => d.pdf_id !== doc.pdf_id));
    } catch (e) {
      console.error('Failed to delete document', e);
      setStatus(`✗ Failed to remove ${doc.filename}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-3">
        <label className={`cursor-pointer px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
          uploading
            ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
            : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
        }`}>
          {uploading ? 'Uploading...' : '📎 Upload PDFs'}
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            multiple
            onChange={handleUpload}
            disabled={uploading}
            className="hidden"
          />
        </label>
        {status && (
          <span className="text-xs text-gray-400 max-w-xs truncate">{status}</span>
        )}
      </div>

      {uploads.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {uploads.map((doc, i) => (
            <div
              key={i}
              className="relative group flex items-center gap-2 px-2 py-1 bg-gray-800 rounded-lg text-xs"
            >
              {/* Deleting spinner */}
              {deletingId === doc.pdf_id ? (
                <span className="text-gray-500 animate-pulse">removing...</span>
              ) : confirmingId === doc.pdf_id ? (
                /* Inline confirmation — no browser popup */
                <div className="flex items-center gap-1.5">
                  <span className="text-gray-400 text-xs">Remove?</span>
                  <button
                    onClick={() => handleDeleteConfirm(doc)}
                    className="px-1.5 py-0.5 bg-red-600 hover:bg-red-500 text-white rounded text-xs font-medium transition-colors"
                  >
                    Yes
                  </button>
                  <button
                    onClick={() => setConfirmingId(null)}
                    className="px-1.5 py-0.5 bg-gray-600 hover:bg-gray-500 text-gray-200 rounded text-xs font-medium transition-colors"
                  >
                    No
                  </button>
                </div>
              ) : (
                /* Normal chip view */
                <>
                  <span className="text-green-400">✓</span>
                  <span className="text-gray-300 truncate max-w-[120px]">{doc.filename}</span>
                  <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                    doc.doc_type === 'research' ? 'bg-blue-900 text-blue-300' :
                    doc.doc_type === 'legal' ? 'bg-purple-900 text-purple-300' :
                    doc.doc_type === 'financial' ? 'bg-green-900 text-green-300' :
                    'bg-gray-700 text-gray-400'
                  }`}>
                    {doc.doc_type}
                  </span>
                  {doc.chunks && (
                    <span className="text-gray-500">{doc.chunks} chunks</span>
                  )}
                </>
              )}

              {/* × button — top right, visible on hover, hidden during confirm/delete */}
              {confirmingId !== doc.pdf_id && deletingId !== doc.pdf_id && (
                <button
                  onClick={() => setConfirmingId(doc.pdf_id)}
                  className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-gray-600 hover:bg-red-500 text-white text-xs items-center justify-center transition-colors opacity-0 group-hover:opacity-100 flex"
                  title="Remove this document"
                >
                  ×
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}