import { X, FileText } from "lucide-react";

export default function CitationModal({ citation, onClose }) {
  // Clean filename: strip UUIDs like "Document (a5d3f68e...)" → show real name or fallback
  const displayFilename = () => {
    const name = citation.filename;
    if (!name || name === "—") return "Unknown Document";
    // If it looks like a raw UUID (no extension, no spaces), show truncated
    if (/^[0-9a-f-]{36}$/i.test(name)) return `Document (${name.slice(0, 8)}...)`;
    return name;
  };

  // Clean excerpt: find first capital letter to avoid mid-sentence starts
  const displayExcerpt = () => {
    const raw = citation.excerpt || citation.text || "";
    if (!raw) return "—";
    // Find first sentence boundary (capital after period, or just first capital)
    const sentenceStart = raw.search(/(?<=[.!?]\s)[A-Z]|^[A-Z]/);
    const cleaned = sentenceStart > 0 ? raw.slice(sentenceStart) : raw;
    // Trim to reasonable length and add ellipsis if cut
    return cleaned.length < raw.length
      ? cleaned.trim() + (cleaned.length < raw.length ? "…" : "")
      : cleaned.trim();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/30 px-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-card border border-border rounded-xl shadow-elevated w-full max-w-md p-6 animate-fade-in-up"
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-foreground font-semibold">
            <FileText className="w-4 h-4 text-primary" />
            Source Citation
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg hover:bg-muted transition-colors"
          >
            <X className="w-4 h-4 text-muted-foreground" />
          </button>
        </div>

        <div className="space-y-3">
          {/* Filename */}
          <div className="p-3 rounded-lg bg-muted">
            <p className="text-xs text-muted-foreground mb-0.5">Document</p>
            <p className="text-sm font-medium text-foreground truncate" title={citation.filename}>
              {displayFilename()}
            </p>
          </div>

          {/* Page */}
          <div className="p-3 rounded-lg bg-muted">
            <p className="text-xs text-muted-foreground mb-0.5">Page</p>
            <p className="text-sm font-medium text-foreground">
              {citation.page ?? citation.page_number ?? "—"}
            </p>
          </div>

          {/* Excerpt */}
          <div className="p-3 rounded-lg bg-muted">
            <p className="text-xs text-muted-foreground mb-0.5">Excerpt</p>
            <p className="text-sm text-foreground leading-relaxed">
              {displayExcerpt()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}