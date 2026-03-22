// api.js — Frontend API layer
// Location: src/lib/api.js

const BASE_URL = "http://localhost:8000";

// ── Auth helpers ──────────────────────────────────────────────────────────────

function getToken() {
  return localStorage.getItem("token");
}

function authHeaders() {
  const token = getToken();
  return token
    ? { "Content-Type": "application/json", Authorization: `Bearer ${token}` }
    : { "Content-Type": "application/json" };
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function signupUser(name, email, password) {
  const res = await fetch(`${BASE_URL}/auth/signup`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ name, email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Signup failed");
  if (data.token) localStorage.setItem("token", data.token);
  return data;
}

export async function loginUser(email, password) {
  const res = await fetch(`${BASE_URL}/auth/login`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ email, password }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed");
  if (data.token) localStorage.setItem("token", data.token);
  return data;
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export async function createSessionOnBackend() {
  try {
    const res = await fetch(`${BASE_URL}/sessions/`, {
      method:  "POST",
      headers: authHeaders(),
    });
    const data = await res.json();
    return data.session_id || null;
  } catch {
    return null;
  }
}

export async function fetchSessions() {
  try {
    const res = await fetch(`${BASE_URL}/sessions/`, { headers: authHeaders() });
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

export async function fetchSessionMessages(sessionId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}/messages`, {
      headers: authHeaders(),
    });
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

export async function deleteSession(sessionId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, {
      method:  "DELETE",
      headers: authHeaders(),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Documents ─────────────────────────────────────────────────────────────────

export async function fetchDocuments(sessionId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}/documents`, {
      headers: authHeaders(),
    });
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

export async function uploadPDF(file, sessionId) {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) formData.append("session_id", sessionId);

  const token = getToken();
  const headers = token ? { Authorization: `Bearer ${token}` } : {};

  const res = await fetch(`${BASE_URL}/upload`, {
    method:  "POST",
    headers,
    body:    formData,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Upload failed");
  return data;
}

export async function checkUploadStatus(jobId) {
  const res = await fetch(`${BASE_URL}/status/${jobId}`, {
    headers: authHeaders(),
  });
  return res.ok ? await res.json() : { status: "error" };
}

export async function deleteDocument(sessionId, docId) {
  try {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}/documents/${docId}`, {
      method:  "DELETE",
      headers: authHeaders(),
    });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Streaming QA ──────────────────────────────────────────────────────────────
//
// Enhancement #16 — responseMode parameter added.
// Sent to backend as `response_mode` field in request body.
// Backend _build_prompt_config uses it to format the answer.
//
// Modes: null (auto) | "short" | "explanation" | "bullets" | "verbatim"
//
// Smooth streaming fix:
//   The onToken callback is called for every SSE token delta.
//   ChatWindow.jsx batches these with a 30ms timer instead of
//   updating React state on every single token — eliminates jitter.

export function streamAnswer(
  question,
  sessionId,
  onToken,
  onDone,
  onError,
  onCitations,
  onMeta,
  responseMode = null,   // Enhancement #16
) {
  const token = getToken();
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  fetch(`${BASE_URL}/qa/ask/stream`, {
    method:  "POST",
    headers,
    body:    JSON.stringify({
      session_id:    sessionId,
      question,
      response_mode: responseMode,   // sent to backend
      user_instruction: null,
    }),
  })
    .then((res) => {
      if (!res.ok) {
        return res.json().then((d) => {
          throw new Error(d.detail || `HTTP ${res.status}`);
        });
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      function read() {
        reader.read().then(({ done, value }) => {
          if (done) {
            onDone();
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop(); // keep incomplete line in buffer

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const payload = JSON.parse(line.slice(6));

              if (payload.type === "token") {
                onToken(payload.content);

              } else if (payload.type === "done") {
                if (payload.citations) onCitations(payload.citations);
                onMeta({
                  query_intent:         payload.query_intent,
                  cache_hit:            payload.cache_hit,
                  hyde_used:            payload.hyde_used,
                  retrieval_confidence: payload.retrieval_confidence,
                });
                onDone();
                return;

              } else if (payload.type === "error") {
                onError(payload.content);
                return;
              }
            } catch {
              // Malformed SSE line — skip
            }
          }

          read(); // continue reading
        });
      }

      read();
    })
    .catch((err) => onError(err.message));
}