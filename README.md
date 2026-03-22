# MultiPDF Agentic AI — Question Answering System

An intelligent document Q&A system that lets you upload multiple PDFs and ask questions in natural language. Built with a full agentic AI pipeline, hybrid retrieval, and real-time streaming responses.

---

## Features

- **Multi-PDF Support** — Upload and query multiple PDFs simultaneously
- **Agentic AI Pipeline** — LLM autonomously decides which tools to use (search, calculate, summarize)
- **Hybrid Search** — Vector search + BM25 keyword search merged via Reciprocal Rank Fusion
- **Real-time Streaming** — Token-by-token streaming responses like ChatGPT
- **Smart Caching** — Semantic cache avoids redundant LLM calls
- **Response Modes** — Auto / Short / Explain / Bullets / Verbatim
- **Document Types** — Auto-detects Legal, Financial, Research, General documents
- **Multi-Provider LLM** — Groq → Cerebras → Together → Gemini → OpenAI fallback chain
- **Memory** — Conversation history with LLM-based compression
- **Citations** — Every answer includes source PDF and page number
- **Auth** — JWT-based signup/login

---

## Tech Stack

### Backend
| Component | Technology |
|-----------|-----------|
| API | FastAPI |
| Pipeline | LangGraph (agentic) |
| Vector DB | ChromaDB |
| SQL DB | PostgreSQL |
| Task Queue | Celery + Redis |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Intent | cross-encoder/nli-MiniLM2-L6-H768 |
| Query Expansion | humarin/chatgpt_paraphraser_on_T5_base |
| LLM | Groq llama-3.3-70b-versatile |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React + Vite |
| Language | TypeScript / JSX |
| Styling | Tailwind CSS |
| Theme | next-themes (dark/light) |

---

## Architecture

```
User Question
      ↓
Input Validation
      ↓
Intent Classification (NLI → Keywords fallback)
      ↓
Coreference Resolution (pronoun rewriting)
      ↓
Query Expansion (T5 — 4 variants)
      ↓
Hybrid Retrieval (Vector + BM25 + RRF)
      ↓
Cross-Encoder Reranking + Threshold Filter
      ↓
┌─────────────────────────────────┐
│  Agentic Generate (summary/     │
│  comparison) — LLM decides      │
│  which tools to call:           │
│  • search_document()            │
│  • get_page()                   │
│  • calculate()                  │
│  • summarize_document()         │
└─────────────────────────────────┘
      OR
┌─────────────────────────────────┐
│  Direct Generate (factual)      │
│  Single LLM call                │
└─────────────────────────────────┘
      ↓
Sufficiency Check (retry if poor)
      ↓
Citations + Semantic Cache Store
      ↓
Streaming Response to Frontend
      ↓
Background Faithfulness Scoring
```

---

## Project Structure

```
Multi-PDF Question Answering System/
│
├── backend/
│   ├── api/
│   │   ├── auth.py              # JWT auth endpoints
│   │   ├── qa.py                # Q&A streaming endpoint
│   │   ├── sessions.py          # Session management
│   │   ├── input_validator.py   # Request validation
│   │   └── pdf_validator.py     # PDF file validation
│   │
│   ├── core/
│   │   ├── embedder.py          # BAAI BGE embeddings (GPU)
│   │   ├── vector_store.py      # ChromaDB + hybrid search
│   │   ├── bm25_store.py        # BM25 keyword index
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   ├── semantic_cache.py    # Cosine similarity cache
│   │   ├── hyde.py              # Hypothetical Document Embeddings
│   │   ├── classifier.py        # Document type classifier
│   │   ├── citations_builder.py # Citation resolver
│   │   ├── query_expander.py    # T5 query expansion
│   │   └── tools.py             # Agent tools
│   │
│   ├── graph/
│   │   ├── graph.py             # LangGraph pipeline
│   │   ├── nodes.py             # Pipeline nodes
│   │   ├── agent_node.py        # Agentic tool-use node
│   │   ├── router.py            # Intent classifier
│   │   ├── coref.py             # Coreference resolution
│   │   ├── state.py             # Pipeline state
│   │   ├── model_router.py      # Doc-type LLM config
│   │   └── fallback_llm.py      # Multi-provider chain
│   │
│   ├── ingestion/
│   │   ├── extractor.py         # PDF text extraction
│   │   └── chunker.py           # Structure-aware chunking
│   │
│   ├── memory/
│   │   ├── chat_history.py      # Message persistence
│   │   ├── context_builder.py   # Memory summarization
│   │   └── session_store.py     # Session management
│   │
│   ├── models/
│   │   ├── database.py          # SQLAlchemy setup
│   │   └── schema.py            # DB models
│   │
│   ├── pipeline/
│   │   └── evaluator.py         # Faithfulness scoring
│   │
│   ├── workers/
│   │   └── pdf_worker.py        # Celery PDF processor
│   │
│   ├── config.py                # Centralized config
│   ├── main.py                  # FastAPI app entry
│   ├── celery_config.py         # Celery setup
│   └── requirements.txt
│
├── pdf-genius-main/             # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx   # Streaming chat
│   │   │   ├── MessageBubble.jsx # Message renderer
│   │   │   ├── InputBar.jsx     # Input + mode selector
│   │   │   ├── UploadZone.jsx   # PDF upload
│   │   │   ├── Sidebar.jsx      # Session list
│   │   │   └── Header.jsx
│   │   ├── context/
│   │   │   └── AppContext.jsx   # Global state
│   │   ├── lib/
│   │   │   └── api.js           # API client
│   │   └── pages/
│   │       ├── Dashboard.jsx
│   │       ├── Login.jsx
│   │       ├── Signup.jsx
│   │       └── LandingPage.jsx
│   └── package.json
│
├── start_all.bat                # Start all services
├── start_backend.bat
├── start_celery.bat
├── start_frontend.bat
├── COMMANDS.md                  # All useful commands
├── .env.example                 # Environment template
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+
- NVIDIA GPU (recommended) or CPU

### 1. Clone Repository
```bash
git clone https://github.com/devloperSai/multiPDF-AgenticAI-final-.git
cd multiPDF-AgenticAI-final-
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
```bash
cp .env.example .env
# Edit .env with your actual values
```

### 4. Database Setup
```bash
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE pdfqa;"

# Add doc_summary column
psql -U postgres -d pdfqa -c "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_summary TEXT;"
```

### 5. Frontend Setup
```bash
cd pdf-genius-main
npm install
```

### 6. Start All Services
```bash
# Windows — double click
start_all.bat

# Or manually:
# Terminal 1
celery -A celery_config.celery_app worker --loglevel=info --pool=solo

# Terminal 2
uvicorn main:app --reload

# Terminal 3
cd ../pdf-genius-main && npm run dev
```

### 7. Open Browser
```
http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/signup | Create account |
| POST | /auth/login | Login |
| GET | /auth/me | Get current user |
| POST | /sessions/ | Create session |
| GET | /sessions/ | List sessions |
| DELETE | /sessions/{id} | Delete session |
| POST | /upload | Upload PDF |
| GET | /status/{job_id} | Check upload status |
| POST | /qa/ask/stream | Ask question (streaming) |
| GET | /health | Health check |

---

## LLM Provider Chain

```
Groq (primary, free 100k tokens/day)
  ↓ if rate limited
Together AI (fallback)
  ↓ if unavailable
Cerebras (free, confirmed working)
  ↓ if unavailable
Gemini (free 1M tokens/day)
  ↓ if unavailable
OpenAI (paid, if key provided)
  ↓ if unavailable
Anthropic (paid, if key provided)
```

Add API key to `.env` to activate a provider. Providers without keys are skipped automatically.

---

## Enhancements Implemented

1. NLI Intent Router — GPU-based zero-shot classification
2. Spread Retrieval for Summary — page-spread sampling
3. Reranker Threshold + Minimum Chunks Fallback
4. Chunk Deduplication — Jaccard similarity
5. Answer Confidence Gating
6. Summary Memory — LLM chat history compression
7. HyDE — Hypothetical Document Embeddings
8. N-Provider Fallback Chain
9. Per-doc-type LLM configs
10. Local Faithfulness Evaluator — cross-encoder scoring
11. Centralized Config — env-overridable
12. Coreference Resolution — pronoun rewriting
13. Response Mode Selector — Short/Explain/Bullets/Verbatim
14. Query Expansion — T5 paraphrase model, 4 variants
15. Session Doc Summarization — ingestion-time summary
16. Frontend Response Mode UI + Smooth Streaming

---

## License

MIT License — free to use for educational and personal projects.

---

## Author

Built by Sai — Computer Science Student
GitHub: https://github.com/devloperSai
