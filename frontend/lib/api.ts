import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000',
});

export const createSession = () =>
  api.post('/sessions/').then(r => r.data);

export const listSessions = () =>
  api.get('/sessions/').then(r => r.data);

export const uploadPDF = (file: File, sessionId: string) => {
  const form = new FormData();
  form.append('file', file);
  form.append('session_id', sessionId);
  return api.post('/upload', form).then(r => r.data);
};

export const pollStatus = (jobId: string) =>
  api.get(`/status/${jobId}`).then(r => r.data);

export const askQuestion = (sessionId: string, question: string) =>
  api.post('/qa/ask', { session_id: sessionId, question }).then(r => r.data);

export const getMessages = (sessionId: string) =>
  api.get(`/sessions/${sessionId}/messages`).then(r => r.data);

export const getDocuments = (sessionId: string) =>
  api.get(`/sessions/${sessionId}/documents`).then(r => r.data);

// Refine 16 — delete a single document from a session
// Cleans: ChromaDB chunks + PostgreSQL record + semantic cache
export const deleteDocument = (sessionId: string, docId: string) =>
  api.delete(`/sessions/${sessionId}/documents/${docId}`).then(r => r.data);

export const askStream = (
  sessionId: string,
  question: string,
  onToken: (token: string) => void,
  onDone: (metadata: any) => void,
  onError: (error: string) => void
) => {
  fetch('http://localhost:8000/qa/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, question })
  }).then(async (response) => {
    if (!response.ok) {
      // Refine 6 — extract actual detail message from FastAPI error response
      try {
        const errorBody = await response.json();
        const detail = errorBody?.detail || `Request failed (${response.status})`;
        onError(detail);
      } catch {
        onError(`Request failed (${response.status})`);
      }
      return;
    }

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      onError('No response body');
      return;
    }

    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'token') {
              onToken(data.content);
            } else if (data.type === 'done') {
              onDone(data);
            } else if (data.type === 'error') {
              onError(data.content);
            }
          } catch (e) {
            // skip malformed chunks
          }
        }
      }
    }
  }).catch(err => onError(err.message));
};

export const getDocumentInfo = (sessionId: string) =>
  api.get(`/sessions/${sessionId}/documents`).then(r => r.data);