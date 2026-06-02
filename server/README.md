# AI Codebase Q&A Tool

This FastAPI app clones a GitHub repo, chunks the code, stores embeddings in ChromaDB, and answers questions from retrieved code context.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- `GET /` - landing page
- `POST /api/ingest` - clone and index a repo
- `POST /api/ask` - ask a question about an indexed repo
- `GET /workflow?repo_url=...` - HTML workflow view
- `GET /api/repo-summary?repo_url=...` - repo stats and file tree preview
