'use client';
import { useState, useEffect } from 'react';
import SessionPanel from './components/SessionPanel';
import UploadPanel from './components/UploadPanel';
import ChatPanel from './components/ChatPanel';

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [uploadedDocs, setUploadedDocs] = useState<string[]>([]);

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Left sidebar — sessions */}
      <div className="w-64 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-lg font-bold text-white">Multi-PDF QA</h1>
          <p className="text-xs text-gray-400 mt-1">Ask questions across documents</p>
        </div>
        <SessionPanel
          activeSessionId={sessionId}
          onSessionSelect={setSessionId}
        />
      </div>

      {/* Main area */}
      <div className="flex-1 flex flex-col">
        {sessionId ? (
          <>
            {/* Upload bar */}
            <div className="border-b border-gray-800 p-3">
              <UploadPanel
                sessionId={sessionId}
                onUploadComplete={(filename) =>
                  setUploadedDocs(prev => [...prev, filename])
                }
              />
            </div>
            {/* Chat */}
            <div className="flex-1 overflow-hidden">
              <ChatPanel sessionId={sessionId} />
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-2xl font-semibold text-gray-400">
                Select or create a session
              </p>
              <p className="text-gray-600 mt-2 text-sm">
                to start uploading PDFs and asking questions
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}