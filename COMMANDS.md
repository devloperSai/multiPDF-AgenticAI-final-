# Multi-PDF QA System — Server Commands

## Quick Start (use .bat files)
Double-click `start_all.bat` to start everything at once.

---

## Manual Commands

### Prerequisites — Start Redis First
```powershell
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Start Redis (if not running)
redis-server
```

---

### Terminal 1 — Celery Worker
```powershell
cd "C:\Multi-PDF Question Answering System\backend"
venv\Scripts\activate
celery -A celery_config.celery_app worker --loglevel=info --pool=solo
```

### Terminal 2 — FastAPI Backend
```powershell
cd "C:\Multi-PDF Question Answering System\backend"
venv\Scripts\activate
uvicorn main:app --reload
```

### Terminal 3 — Frontend
```powershell
cd "C:\Multi-PDF Question Answering System\pdf-genius-main"
npm run dev
```

---

## URLs
| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:5173        |
| Backend   | http://localhost:8000        |
| API Docs  | http://localhost:8000/docs   |
| Redis     | localhost:6379               |

---

## Useful Commands

### Check what's running
```powershell
# Check Redis
redis-cli ping

# Check FastAPI
curl http://localhost:8000/health

# Check all ports in use
netstat -ano | findstr "8000\|5173\|6379"
```

### Stop a port if stuck
```powershell
# Find process on port 8000
netstat -ano | findstr :8000

# Kill it (replace XXXX with PID)
taskkill /PID XXXX /F
```

### Restart FastAPI only
```powershell
# Press Ctrl+C in FastAPI terminal, then:
uvicorn main:app --reload
```

### Clear Redis cache (reset Celery queue)
```powershell
redis-cli FLUSHALL
```

### Check Celery is receiving tasks
```powershell
celery -A celery_config.celery_app inspect active
```

---

## Install / Setup (first time only)

### Backend
```powershell
cd "C:\Multi-PDF Question Answering System\backend"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```powershell
cd "C:\Multi-PDF Question Answering System\pdf-genius-main"
npm install
```

### Database
```powershell
# Create PostgreSQL database
psql -U postgres -c "CREATE DATABASE pdfqa;"

# Add doc_summary column (run once)
psql -U postgres -d pdfqa -c "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_summary TEXT;"
```

---

## Environment Variables (.env)
```
# Required
DATABASE_URL=postgresql://postgres:password@localhost/pdfqa
GROQ_API_KEY=your_groq_key
JWT_SECRET=your_jwt_secret
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Fallback LLM providers (add as needed)
TOGETHER_API_KEY=your_together_key
CEREBRAS_API_KEY=your_cerebras_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Provider Fallback Chain
```
Groq → Together → Cerebras → Gemini → OpenAI → Anthropic
```
- Groq:     Free, 100k tokens/day — PRIMARY
- Cerebras: Free, confirmed working — BACKUP
- Others:   Add API key to .env to activate

---

## Common Errors

| Error | Fix |
|-------|-----|
| `redis connection refused` | Start Redis first |
| `port 8000 already in use` | Kill process on port 8000 |
| `groq 429 rate limit` | Wait for reset or Cerebras takes over |
| `celery task not received` | Check Redis is running |
| `CUDA out of memory` | Restart Celery worker |
