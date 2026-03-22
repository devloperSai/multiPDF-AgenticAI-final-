import { useState, useRef, useEffect } from "react";
import { Upload, FileText, CheckCircle, Loader2, AlertCircle, X } from "lucide-react";
import { uploadPDF, checkUploadStatus, deleteDocument, fetchDocuments } from "../lib/api";
import { useApp } from "../context/AppContext";

const STAGES = ["extracting", "classifying", "chunking", "embedding", "storing", "success"];

const DOC_TYPE_STYLES = {
  research:  { label: "Research",  cls: "bg-blue-500/15 text-blue-400 border border-blue-500/25" },
  legal:     { label: "Legal",     cls: "bg-amber-500/15 text-amber-400 border border-amber-500/25" },
  financial: { label: "Financial", cls: "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" },
  general:   { label: "General",   cls: "bg-muted text-muted-foreground border border-border" },
};

function DocTypeBadge({ docType }) {
  if (!docType) return null;
  const s = DOC_TYPE_STYLES[docType] || DOC_TYPE_STYLES.general;
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide shrink-0 ${s.cls}`}>
      {s.label}
    </span>
  );
}

export default function UploadZone() {
  const { activeSessionId, createSession } = useApp();
  const [dragOver, setDragOver] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(false);

  // docs shape:
  // { docId, filename, docType, stage, error, confirmDelete, deleting }
  // stage: null = done/idle, "uploading"|"extracting"|... = in progress, "success" = complete
  const [docs, setDocs] = useState([]);

  const fileRef = useRef();

  // ── Fetch existing docs whenever active session changes ───────────────────
  useEffect(() => {
    if (!activeSessionId) {
      setDocs([]);
      return;
    }

    setLoadingDocs(true);
    fetchDocuments(activeSessionId)
      .then((data) => {
        if (!Array.isArray(data)) return;
        // Backend returns: [{ filename, doc_type, pdf_id, status }]
        const mapped = data.map((d) => ({
          docId:         d.pdf_id,
          filename:      d.filename,
          docType:       d.doc_type || "general",
          stage:         "success",   // already processed docs
          error:         null,
          confirmDelete: false,
          deleting:      false,
        }));
        setDocs(mapped);
      })
      .catch(() => setDocs([]))
      .finally(() => setLoadingDocs(false));

  }, [activeSessionId]);

  // ── helpers ───────────────────────────────────────────────────────────────

  const setDocField = (docId, fields) =>
    setDocs((prev) => prev.map((d) => d.docId === docId ? { ...d, ...fields } : d));

  // ── upload one file ───────────────────────────────────────────────────────

  const uploadOne = async (file, sessionId) => {
    if (!file.name.endsWith(".pdf")) {
      const tempId = "err_" + Date.now() + "_" + file.name;
      setDocs((prev) => [...prev, {
        docId: tempId, filename: file.name, docType: null,
        stage: null, error: "Not a PDF file", confirmDelete: false, deleting: false,
      }]);
      return;
    }

    const tempId = "pending_" + Date.now() + "_" + file.name;
    setDocs((prev) => [...prev, {
      docId: tempId, filename: file.name, docType: null,
      stage: "uploading", error: null, confirmDelete: false, deleting: false,
    }]);

    try {
      const data = await uploadPDF(file, sessionId);
      const { job_id, doc_id } = data;

      // Swap tempId → real doc_id from backend
      setDocs((prev) =>
        prev.map((d) => d.docId === tempId ? { ...d, docId: doc_id, stage: "extracting" } : d)
      );

      pollStatus(job_id, doc_id);
    } catch (e) {
      setDocField(tempId, { stage: null, error: e.message || "Upload failed" });
    }
  };

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return;
    let sessionId = activeSessionId;
    if (!sessionId) sessionId = await createSession();
    Array.from(files).forEach((file) => uploadOne(file, sessionId));
  };

  // ── poll one job ──────────────────────────────────────────────────────────

  const pollStatus = (jobId, docId) => {
    const interval = setInterval(async () => {
      try {
        const data = await checkUploadStatus(jobId);
        const stage = data.stage || data.status;
        setDocField(docId, { stage, error: null });

        if (stage === "success") {
          clearInterval(interval);
          if (data.doc_type) setDocField(docId, { stage: "success", docType: data.doc_type });
        }
        if (stage === "error" || stage === "failed") {
          clearInterval(interval);
          setDocField(docId, { stage: null, error: data.message || "Processing failed" });
        }
      } catch {
        clearInterval(interval);
        setDocField(docId, { stage: null, error: "Status check failed" });
      }
    }, 2000);
  };

  // ── drag & drop ───────────────────────────────────────────────────────────

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  // ── delete handlers ───────────────────────────────────────────────────────

  const handleDeleteConfirm  = (docId) => setDocField(docId, { confirmDelete: true });
  const handleDeleteCancel   = (docId) => setDocField(docId, { confirmDelete: false });
  const handleDeleteExecute  = async (docId) => {
    setDocField(docId, { deleting: true });
    await deleteDocument(activeSessionId, docId);
    setDocs((prev) => prev.filter((d) => d.docId !== docId));
  };

  // ── inline stage progress ─────────────────────────────────────────────────

  const DocStages = ({ stage }) => {
    if (!stage || stage === "success") return null;
    return (
      <div className="flex flex-col gap-0.5 mt-1.5">
        {STAGES.map((s) => {
          const idx    = STAGES.indexOf(s);
          const curIdx = STAGES.indexOf(stage);
          const done   = idx < curIdx;
          const active = idx === curIdx;
          return (
            <div key={s} className={`flex items-center gap-1.5 text-[10px] capitalize ${
              done ? "text-primary" : active ? "text-foreground" : "text-muted-foreground/50"
            }`}>
              {done   ? <CheckCircle className="w-2.5 h-2.5" />
               : active ? <Loader2 className="w-2.5 h-2.5 animate-spin" />
               : <span className="w-2.5 h-2.5 text-center leading-none">·</span>}
              {s}
            </div>
          );
        })}
      </div>
    );
  };

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="p-4 flex flex-col gap-3">
      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Documents</p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          dragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground"
        }`}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <Upload className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
        <p className="text-sm text-foreground font-medium">Drop PDFs here or click to upload</p>
        <p className="text-xs text-muted-foreground mt-1">Multiple PDFs supported · Max 50MB each</p>
      </div>

      {/* Doc list */}
      {loadingDocs ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground py-1">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          Loading documents...
        </div>
      ) : docs.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground font-medium">
            {docs.length} document{docs.length !== 1 ? "s" : ""}
          </p>

          {docs.map((doc) => (
            <div key={doc.docId} className="group relative flex flex-col px-3 py-2 rounded-lg bg-muted text-xs">
              {doc.confirmDelete ? (
                <div className="flex items-center gap-2">
                  <span className="text-destructive font-medium">Remove?</span>
                  <button
                    onClick={() => handleDeleteExecute(doc.docId)}
                    disabled={doc.deleting}
                    className="px-2 py-0.5 rounded bg-destructive text-destructive-foreground font-medium hover:opacity-80 transition-opacity"
                  >
                    {doc.deleting ? "..." : "Yes"}
                  </button>
                  <button
                    onClick={() => handleDeleteCancel(doc.docId)}
                    className="px-2 py-0.5 rounded bg-muted-foreground/20 text-foreground hover:opacity-80 transition-opacity"
                  >
                    No
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-2">
                    <FileText className="w-3.5 h-3.5 text-primary shrink-0" />
                    <span className="truncate flex-1 text-foreground">{doc.filename}</span>
                    {doc.stage === "success" && <DocTypeBadge docType={doc.docType} />}
                    {doc.stage === "success" && (
                      <button
                        onClick={() => handleDeleteConfirm(doc.docId)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:bg-destructive/20 shrink-0"
                      >
                        <X className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                      </button>
                    )}
                  </div>

                  {doc.error && (
                    <div className="flex items-center gap-1.5 mt-1.5 text-destructive">
                      <AlertCircle className="w-3 h-3 shrink-0" />
                      <span>{doc.error}</span>
                    </div>
                  )}

                  <DocStages stage={doc.stage} />
                </>
              )}
            </div>
          ))}
        </div>
      ) : activeSessionId ? (
        <p className="text-xs text-muted-foreground/50 text-center py-2">No documents in this session</p>
      ) : null}
    </div>
  );
}