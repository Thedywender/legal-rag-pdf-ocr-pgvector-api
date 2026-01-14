# Legal RAG API (PDF + OCR + pgvector) — Flask + Docker Compose

API de demonstração de RAG para consultar PDFs (inclusive escaneados), com fallback de OCR por página.

## Stack

- Flask (API)
- PyMuPDF (extração de texto e renderização de páginas)
- Tesseract OCR (pytesseract)
- LangChain (Document, splitter, PGVector)
- Postgres + pgvector (armazenamento vetorial)
- OpenAI embeddings + chat model

## Como rodar (Docker Compose)

### 1) Variáveis de ambiente

Crie `.env` a partir do exemplo:

```bash
cp .env.example .env
```
