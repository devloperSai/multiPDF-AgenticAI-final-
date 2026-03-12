'use client';
import { useState, useEffect, useRef } from 'react';
import { askStream, getMessages } from '../../lib/api';
import CitationPanel from './CitationPanel';

interface Citation {
  source_index: number;
  pdf_id: string;
  chunk_index: number;
  page_number?: string | null;
  score: number;
  excerpt: string;
  filename?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  created_at?: string;
  streaming?: boolean;
  citations?: Citation[];
}

interface Props {
  sessionId: string;
}

function renderMessageWithCitations(
  content: string,
  citations: Citation[],
  onCitationClick: (c: Citation) => void
) {
  // Replace [Source N] with clickable chips
  const parts = content.split(/(\[Source \d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/\[Source (\d+)\]/);
    if (match) {
      const sourceNum = parseInt(match[1]);
      const citation = citations.find(c => c.source_index === sourceNum);
      if (citation) {
        return (
          <button
            key={i}
            onClick={() => onCitationClick(citation)}
            className="inline-flex items-center px-1.5 py-0.5 mx-0.5 bg-blue-900/60 hover:bg-blue-800 text-blue-300 rounded text-xs font-medium transition-colors cursor-pointer border border-blue-800/50"
          >
            {part}
          </button>
        );
      }
    }
    return <span key={i}>{part}</span>;
  });
}

export default function ChatPanel({ sessionId }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const loadMessages = async () => {
      try{
        const data = await getMessages(sessionId);
        setMessages(data.map((m: any) => ({
          role: m.role,
          content: m.content,
          created_at: m.created_at,
          citations: m.citations || []
        })));
      } catch (e) {
        console.error('Failed to load messages', e);
      }
    };
    setMessages([]);
    loadMessages();
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setLoading(true);

    setMessages(prev => [...prev, { role: 'user', content: question }]);
    setMessages(prev => [...prev, { role: 'assistant', content: '', streaming: true, citations: [] }]);

    askStream(
      sessionId,
      question,
      (token) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + token
            };
          }
          return updated;
        });
      },
      (metadata) => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              streaming: false,
              citations: metadata.citations || []
            };
          }
          return updated;
        });
        setLoading(false);
      },
      (error) => {
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'assistant',
            content: `Error: ${error}`,
            streaming: false,
            citations: []
          };
          return updated;
        });
        setLoading(false);
      }
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <div className="flex flex-col h-full">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500 text-sm">
                Upload a PDF and ask a question
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-sm'
                  : 'bg-gray-800 text-gray-100 rounded-bl-sm'
              }`}>
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && !msg.streaming ? (
                  <p className="whitespace-pre-wrap font-sans">
                    {renderMessageWithCitations(msg.content, msg.citations, setActiveCitation)}
                  </p>
                ) : (
                  <pre className="whitespace-pre-wrap font-sans">{msg.content}</pre>
                )}
                {msg.streaming && (
                  <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5 rounded align-middle"/>
                )}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-800 p-4">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your documents..."
              rows={1}
              className="flex-1 bg-gray-800 text-gray-100 rounded-xl px-4 py-2.5 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-blue-500 placeholder-gray-500"
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-medium transition-colors"
            >
              {loading ? 'Sending...' : 'Send'}
            </button>
          </div>
          <p className="text-xs text-gray-600 mt-1.5 ml-1">Enter to send · Shift+Enter for new line</p>
        </div>
      </div>

      {/* Citation Panel */}
      <CitationPanel
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />
    </>
  );
}