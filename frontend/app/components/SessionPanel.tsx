'use client';
import { useState, useEffect } from 'react';
import { createSession, listSessions } from '../../lib/api';

interface Session {
  session_id: string;
  title: string | null;
  updated_at: string;
}

interface Props {
  activeSessionId: string | null;
  onSessionSelect: (id: string) => void;
}

export default function SessionPanel({ activeSessionId, onSessionSelect }: Props) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchSessions = async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch (e) {
      console.error('Failed to fetch sessions', e);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const handleCreate = async () => {
    setLoading(true);
    try {
      const data = await createSession();
      await fetchSessions();
      onSessionSelect(data.session_id);
    } catch (e) {
      console.error('Failed to create session', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col p-3 gap-2 overflow-y-auto">
      <button
        onClick={handleCreate}
        disabled={loading}
        className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
      >
        {loading ? 'Creating...' : '+ New Session'}
      </button>

      <div className="flex flex-col gap-1 mt-2">
        {sessions.length === 0 && (
          <p className="text-xs text-gray-500 text-center mt-4">No sessions yet</p>
        )}
        {sessions.map(s => (
          <button
            key={s.session_id}
            onClick={() => onSessionSelect(s.session_id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
              activeSessionId === s.session_id
                ? 'bg-blue-600 text-white'
                : 'hover:bg-gray-800 text-gray-300'
            }`}
          >
            <div className="font-medium truncate">
              {s.title || `Session ${s.session_id.slice(0, 8)}`}
            </div>
            <div className="text-xs text-gray-400 mt-0.5">
              {new Date(s.updated_at).toLocaleDateString()}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}