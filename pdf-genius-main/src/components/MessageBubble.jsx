import { useState } from "react";
import { Bot, User, Zap, FlaskConical, BarChart2, FileText } from "lucide-react";
import CitationModal from "./CitationModal";

// ── Intent / meta badges ──────────────────────────────────────────────────────

const INTENT_STYLES = {
  factual:      { label: "Factual",      cls: "bg-blue-500/10 text-blue-400 border border-blue-500/20" },
  summary:      { label: "Summary",      cls: "bg-violet-500/10 text-violet-400 border border-violet-500/20" },
  comparison:   { label: "Comparison",   cls: "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" },
  out_of_scope: { label: "Out of scope", cls: "bg-muted text-muted-foreground border border-border" },
};

function IntentBadge({ intent }) {
  if (!intent) return null;
  const s = INTENT_STYLES[intent] || INTENT_STYLES.factual;
  return <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${s.cls}`}>{s.label}</span>;
}

function CacheHitBadge({ cacheHit }) {
  if (!cacheHit) return null;
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-amber-500/10 text-amber-400 border border-amber-500/20">
      <Zap className="w-2.5 h-2.5" />Cached
    </span>
  );
}

function HydeBadge({ hydeUsed }) {
  if (!hydeUsed) return null;
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-pink-500/10 text-pink-400 border border-pink-500/20">
      <FlaskConical className="w-2.5 h-2.5" />HyDE
    </span>
  );
}

function ConfidenceBadge({ score }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const cls = pct >= 70
    ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
    : pct >= 40
      ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
      : "bg-red-500/10 text-red-400 border border-red-500/20";
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${cls}`}>
      <BarChart2 className="w-2.5 h-2.5" />{pct}% confidence
    </span>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Shorten a filename for inline display.
 * "Non-Disclosure_Agreement.pdf" → "Non-Disclosure_Agreement.pdf"  (≤22 chars, keep as-is)
 * "A-very-long-document-name.pdf" → "A-very-long-docum….pdf"
 */
function shortName(filename) {
  if (!filename) return "Source";
  const MAX = 22;
  if (filename.length <= MAX) return filename;
  const ext   = filename.includes(".") ? filename.slice(filename.lastIndexOf(".")) : "";
  const base  = filename.slice(0, filename.lastIndexOf("."));
  const trimmed = base.slice(0, MAX - ext.length - 1);
  return `${trimmed}…${ext}`;
}

// ── Markdown + citation renderer ─────────────────────────────────────────────

// Render a single citation chip
function CitationChip({ num, citations, setCitation, keyProp }) {
  const src = citations?.[num - 1];
  const label = src
    ? src.page
      ? `${shortName(src.filename)} · p.${src.page}`
      : shortName(src.filename)
    : `Source ${num}`;
  return (
    <button
      key={keyProp}
      onClick={() => src && setCitation(src)}
      title={src?.filename || `Source ${num}`}
      className="inline-flex items-center gap-0.5 px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors align-baseline"
    >
      <FileText className="w-2.5 h-2.5 shrink-0" />
      <span className="truncate max-w-[120px]">{label}</span>
    </button>
  );
}

function renderInline(text, citations, setCitation) {
  // Split on both single [Source N] and multi [Source 1, Source 2, Source 4]
  const parts = text.split(/(\*\*[^*]+\*\*|\[Source [\d,\s]+\])/g);
  return parts.map((part, i) => {
    // Bold
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-foreground">{part.slice(2, -2)}</strong>;
    }

    // Single citation: [Source N]
    const singleMatch = part.match(/^\[Source (\d+)\]$/);
    if (singleMatch) {
      return <CitationChip key={i} num={parseInt(singleMatch[1])} citations={citations} setCitation={setCitation} />;
    }

    // Multi citation: [Source 1, Source 2, Source 4]
    const multiMatch = part.match(/^\[Source ([\d,\s]+)\]$/);
    if (multiMatch) {
      const nums = multiMatch[1].split(",").map(n => parseInt(n.trim())).filter(Boolean);
      return (
        <span key={i}>
          {nums.map((num, j) => (
            <CitationChip key={j} num={num} citations={citations} setCitation={setCitation} />
          ))}
        </span>
      );
    }

    return <span key={i}>{part}</span>;
  });
}

// Extract [Source N] tags from text, return {clean text, source numbers[]}
function extractSources(text) {
  const sourceNums = [];
  const clean = text.replace(/\[Source (\d+)\]/g, (_, n) => {
    sourceNums.push(parseInt(n));
    return "";
  }).trim();
  return { clean, sourceNums: [...new Set(sourceNums)] };
}

// Compact citation chip — just the page number, smaller, inline
function CompactCitation({ num, citations, setCitation }) {
  const src = citations?.[num - 1];
  if (!src) return null;
  const label = src.page ? `p.${src.page}` : src.filename?.split(".")[0]?.slice(0, 8) || `${num}`;
  return (
    <button
      onClick={() => setCitation(src)}
      title={src.filename}
      className="inline-flex items-center gap-0.5 px-1 py-0.5 mx-0.5 rounded text-[10px] font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors align-baseline"
    >
      <FileText className="w-2.5 h-2.5 shrink-0" />
      <span>{label}</span>
    </button>
  );
}

function MarkdownBody({ text, citations, setCitation }) {
  // Strip [Source N] citations — we show PDF name + page chips instead
  let cleanText = text.replace(/\[Source [\d,\s]+\]/g, "");

  // ── Normalize inline bullet markers to proper newlines ────────────────────
  // LLMs often output everything on one line separated by * or +:
  //   "* South Goa: X * Central Goa: Y" → split into separate lines
  //   "* Section: + Sub1 + Sub2"        → split into indented lines
  cleanText = cleanText.replace(/ \* /g, "\n- ");
  cleanText = cleanText.replace(/^\* /gm, "- ");
  cleanText = cleanText.replace(/ \+ /g, "\n  - ");
  cleanText = cleanText.replace(/^\+ /gm, "  - ");
  cleanText = cleanText.replace(/ {2,}/g, " ");

  const lines = cleanText.split("\n");
  const blocks = [];
  let listType = null;
  let listItems = [];

  const flushList = () => {
    if (listItems.length === 0) return;
    const Tag = listType === "ol" ? "ol" : "ul";
    const cls = listType === "ol"
      ? "list-decimal list-inside space-y-1 my-2 pl-1"
      : "list-disc list-inside space-y-1 my-2 pl-1";
    blocks.push(
      <Tag key={blocks.length} className={cls}>
        {listItems.map((item, i) => {
          // Extract citations from list item text — render as compact chips
          // instead of full-size badges which dominate the list layout
          const { clean, sourceNums } = extractSources(item);
          return (
            <li key={i} className="text-sm leading-relaxed">
              {renderInline(clean, citations, setCitation)}
              {sourceNums.map((n) => (
                <CompactCitation key={n} num={n} citations={citations} setCitation={setCitation} />
              ))}
            </li>
          );
        })}
      </Tag>
    );
    listItems = [];
    listType = null;
  };

  lines.forEach((line, idx) => {
    if (line.startsWith("## ")) {
      flushList();
      blocks.push(
        <h2 key={idx} className="text-base font-semibold text-foreground mt-4 mb-1 first:mt-0">
          {renderInline(line.slice(3).trim(), citations, setCitation)}
        </h2>
      );
      return;
    }
    if (line.startsWith("### ")) {
      flushList();
      blocks.push(
        <h3 key={idx} className="text-sm font-semibold text-foreground mt-3 mb-0.5">
          {renderInline(line.slice(4).trim(), citations, setCitation)}
        </h3>
      );
      return;
    }
    // Handle indented sub-bullets (  - item) — render as nested or flat
    const subMatch = line.match(/^\s{2,}[-*]\s+(.*)/);
    if (subMatch) {
      listType = listType || "ul";
      listItems.push("  " + subMatch[1]);  // prefix with spaces to show nesting
      return;
    }
    const ulMatch = line.match(/^[-*]\s+(.*)/);
    if (ulMatch) {
      if (listType === "ol") flushList();
      listType = "ul";
      listItems.push(ulMatch[1]);
      return;
    }
    const olMatch = line.match(/^\d+\.\s+(.*)/);
    if (olMatch) {
      if (listType === "ul") flushList();
      listType = "ol";
      listItems.push(olMatch[1]);
      return;
    }
    if (line.trim() === "") {
      flushList();
      return;
    }
    flushList();
    blocks.push(
      <p key={idx} className="text-sm leading-relaxed">
        {renderInline(line, citations, setCitation)}
      </p>
    );
  });

  flushList();
  return <div className="space-y-1">{blocks}</div>;
}

// ── Main component ────────────────────────────────────────────────────────────

export default function MessageBubble({ message }) {
  const [citation, setCitation] = useState(null);
  const isUser = message.role === "user";
  const { intent, cacheHit, hydeUsed, retrievalConfidence } = message;
  const hasMeta = intent || cacheHit || hydeUsed;

  return (
    <>
      <div className={`flex gap-3 ${isUser ? "justify-end" : ""}`}>
        {!isUser && (
          <div className="w-7 h-7 rounded-lg gradient-primary flex items-center justify-center shrink-0 mt-0.5">
            <Bot className="w-3.5 h-3.5 text-primary-foreground" />
          </div>
        )}

        <div className="flex flex-col gap-1.5 max-w-[75%]">
          <div className={`px-4 py-3 rounded-2xl ${
            isUser
              ? "bg-primary text-primary-foreground rounded-br-md text-sm leading-relaxed"
              : "bg-chat-ai text-foreground rounded-bl-md"
          }`}>
            {isUser
              ? message.text
              : <MarkdownBody
                  text={message.text || ""}
                  citations={message.citations}
                  setCitation={setCitation}
                />
            }
          </div>

          {!isUser && hasMeta && (
            <div className="flex flex-wrap items-center gap-1.5 pl-1">
              <IntentBadge intent={intent} />
              <CacheHitBadge cacheHit={cacheHit} />
              <HydeBadge hydeUsed={hydeUsed} />
            </div>
          )}
        </div>

        {isUser && (
          <div className="w-7 h-7 rounded-lg bg-muted flex items-center justify-center shrink-0 mt-0.5">
            <User className="w-3.5 h-3.5 text-muted-foreground" />
          </div>
        )}
      </div>

      {citation && <CitationModal citation={citation} onClose={() => setCitation(null)} />}
    </>
  );
}