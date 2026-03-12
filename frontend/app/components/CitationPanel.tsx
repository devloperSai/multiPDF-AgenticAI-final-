'use client';

interface Citation {
  source_index: number;
  pdf_id: string;
  chunk_index: number;
  score: number;
  excerpt: string;
  filename?: string;
  page_number?: string | null;
}

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

export default function CitationPanel({ citation, onClose }: Props) {
  if (!citation) return null;

  const confidenceColor =
    citation.score >= 0.8 ? 'bg-green-500' :
    citation.score >= 0.6 ? 'bg-yellow-500' :
    'bg-red-500';

  const confidenceLabel =
    citation.score >= 0.8 ? 'High' :
    citation.score >= 0.6 ? 'Medium' : 'Low';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-md bg-gray-900 rounded-2xl border border-gray-700 shadow-2xl z-10 overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-400"/>
            <span className="text-white font-medium text-sm">Source {citation.source_index}</span>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Meta info */}
        <div className="px-5 py-3 border-b border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-gray-500 text-xs">📄</span>
              <span className="text-gray-300 text-xs font-medium truncate max-w-[220px]">
                {citation.filename || `Document (${citation.pdf_id.slice(0, 8)}...)`}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${confidenceColor}`}/>
              <span className="text-xs text-gray-400">{confidenceLabel} match</span>
              <span className="text-xs text-gray-600">({Math.round(citation.score * 100)}%)</span>
            </div>
          </div>

          {/* Page number row */}
          <div className="flex items-center gap-4">
            {citation.page_number && citation.page_number !== "" ? (
              <div className="flex items-center gap-1.5">
                <span className="text-gray-500 text-xs">📖</span>
                <span className="text-xs text-gray-400">
                  Page <span className="text-white font-medium">{citation.page_number}</span>
                </span>
              </div>
            ) : (
              <span className="text-xs text-gray-600 italic">Page number unavailable</span>
            )}
            <span className="text-xs text-gray-600">Chunk #{citation.chunk_index}</span>
          </div>
        </div>

        {/* Excerpt */}
        <div className="px-5 py-4">
          <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">
            Referenced passage
          </p>
          <blockquote className="border-l-2 border-blue-600 pl-4">
            <p className="text-sm text-gray-200 leading-relaxed">
              {citation.excerpt}
              {citation.excerpt.length >= 200 && (
                <span className="text-gray-500 italic"> ...continued in document</span>
              )}
            </p>
          </blockquote>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 bg-gray-800/50 flex items-center justify-between">
          <span className="text-xs text-gray-600">
            This passage was used to generate the answer
          </span>
        </div>
      </div>
    </div>
  );
}