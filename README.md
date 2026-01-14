# Legal RAG API (PDF + OCR + pgvector) — Flask + Docker Compose

A portfolio-grade **Retrieval-Augmented Generation (RAG)** demo API that indexes PDFs (including scanned/image-based PDFs) and supports question answering with **source/page traceability**.

This repository is designed to be **safe to publish**:

- No API keys are included.
- No PDFs are committed.
- Everything runs locally with Docker Compose (API + Postgres/pgvector).

---

## What this project does

1. Reads all PDFs from the `docs/` folder
2. For each PDF page:
   - extracts text using **PyMuPDF**
   - if the extracted text is too short, it renders the page as an image and runs **OCR (Tesseract)**
3. Splits text into chunks for retrieval
4. Creates embeddings and stores vectors in **PostgreSQL + pgvector**
5. Exposes a REST API:
   - `GET /health` — health check
   - `POST /qa` — answers questions using retrieved chunks + LLM (RAG)

---

## Tech Stack

### API & Runtime

- **Python 3.10+**
- **Flask** (REST API)
- **python-dotenv** (env loading)

### Document Processing

- **PyMuPDF (fitz)** (PDF text extraction + page rendering)
- **Pillow** (image handling)
- **Tesseract OCR + pytesseract** (OCR fallback)

### RAG / Vector Search

- **LangChain**
  - `Document`
  - `RecursiveCharacterTextSplitter`
  - `PGVector` integration

### Storage

- **PostgreSQL**
- **pgvector** (vector extension)

### DevOps

- **Docker + Docker Compose** (reproducible local setup)

---

## Repository structure

├── src/
│ └── app.py # Flask API + ingestion + RAG pipeline
├── docs/ # Put PDFs here (not committed)
│ └── .gitkeep
├── examples/
│ └── curl_examples.sh # API smoke tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

---

## Important notes (before running)

### 1) About `docs/.gitkeep`

Git does not track empty folders. The `.gitkeep` file exists only to keep the `docs/` directory in the repository.

You do **not** need to add anything inside `.gitkeep`.  
To test ingestion, place your PDFs inside `docs/` (see below).

### 2) Do not commit PDFs or API keys

This project intentionally ignores:

- `.env`
- `docs/*.pdf`

Make sure your `.gitignore` includes:

``gitignore`
.env
docs/\*.pdf

### Run:

`docker compose up --build`

### Health check

`curl http://localhost:5000/health`

### Expected response:

`{"status":"ok","collection":"legal_docs"}`

### QA endpoint (RAG)

`curl -X POST http://localhost:5000/qa \`
-H "Content-Type: application/json" \
 -d '{"question":"Summarize the main obligations."}'
