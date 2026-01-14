import os
import io
import logging
import traceback
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from dotenv import load_dotenv

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

import psycopg2

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------
# Logging (melhor que print)
# ---------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("legal_rag_api")


# -----------------------------
# Config / Environment
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")  # postgresql://user:pass@host:5432/db
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Evita reindexar toda vez sem querer
INGEST_ON_STARTUP = os.getenv("INGEST_ON_STARTUP", "true").lower() == "true"

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment (.env).")
if not DATABASE_URL:
    raise RuntimeError("Missing DATABASE_URL in environment (.env).")

app = Flask(__name__)


# -----------------------------
# 1) Ensure pgvector exists
# -----------------------------
def ensure_pgvector_extension(database_url: str) -> None:
    """
    Ensures pgvector extension exists.
    Note: this requires privileges. In managed Postgres, you may need to enable it in the provider console.
    """
    conn = None
    try:
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension ensured.")
    except Exception as e:
        logger.error("Failed to ensure pgvector extension. Error=%s", e)
        raise
    finally:
        if conn:
            conn.close()


# -----------------------------
# 2) PDF text extraction with OCR fallback
# -----------------------------
def page_to_pil_image(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """
    Renders a PDF page to a PIL image using PyMuPDF (no poppler needed).
    """
    zoom = dpi / 72.0  # 72 dpi is the PDF default
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return img.convert("RGB")  # OCR tende a ser mais estável em RGB


def extract_pdf_documents_with_ocr(
    pdf_path: str, min_text_len: int = 30
) -> List[Document]:
    """
    Extracts per-page text from a PDF.
    If extracted text is too short, uses OCR on that page.
    Returns a list of LangChain Document objects (one per page).
    """
    docs: List[Document] = []
    pdf_name = os.path.basename(pdf_path)

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = (page.get_text("text") or "").strip()

            used_ocr = False
            if len(text) < min_text_len:
                try:
                    img = page_to_pil_image(page, dpi=300)
                    ocr_text = (pytesseract.image_to_string(img) or "").strip()
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        used_ocr = True
                except Exception as e:
                    # Não silencie completamente: ajuda muito no diagnóstico
                    logger.warning(
                        "OCR failed for %s page %s. Error=%s",
                        pdf_name,
                        page_index + 1,
                        e,
                    )

            metadata = {
                "source": pdf_name,
                "path": pdf_path,
                "page": page_index + 1,
                "used_ocr": used_ocr,
            }

            if text:
                docs.append(Document(page_content=text, metadata=metadata))

    return docs


# -----------------------------
# 3) Load all PDFs from docs/
# -----------------------------
def load_all_pdfs(docs_dir: str) -> List[Document]:
    """
    Loads all PDFs from docs_dir and returns a list of per-page Documents.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")

    pdf_files = [
        os.path.join(docs_dir, f)
        for f in os.listdir(docs_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {docs_dir}")

    all_docs: List[Document] = []
    for pdf_path in pdf_files:
        try:
            all_docs.extend(extract_pdf_documents_with_ocr(pdf_path))
            logger.info("Loaded %s", os.path.basename(pdf_path))
        except Exception as e:
            logger.warning("Failed to process %s. Error=%s", pdf_path, e)

    return all_docs


# -----------------------------
# 4) Split into chunks
# -----------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return splitter.split_documents(docs)


# -----------------------------
# 5) Embed + store in PostgreSQL/pgvector
# -----------------------------
def build_vectorstore(chunks: List[Document]) -> PGVector:
    """
    Creates embeddings and stores them in Postgres (pgvector) using PGVector.
    Important: behavior (append vs recreate) depends on langchain version.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=DATABASE_URL,
        # Se sua versão suportar, isso evita duplicação:
        # pre_delete_collection=True,
    )
    return vectorstore


# -----------------------------
# 6) RAG answer function
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a legal-document assistant. Use only the provided context. "
            "If the answer is not in the context, say you don't know.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer clearly and concisely.",
        ),
    ]
)


def answer_with_rag(vectorstore: PGVector, question: str, k: int = 4) -> Dict[str, Any]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(question)

    # Guardrail simples para não explodir contexto
    max_chars = 12_000
    parts: List[str] = []
    total = 0
    for d in retrieved_docs:
        chunk = f"[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]\n{d.page_content}"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    context = "\n\n---\n\n".join(parts)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    msg = RAG_PROMPT.format_messages(question=question, context=context)
    resp = llm.invoke(msg)

    sources = [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "used_ocr": d.metadata.get("used_ocr"),
        }
        for d in retrieved_docs
    ]

    return {"answer": resp.content, "sources": sources}


# -----------------------------
# App startup: ingest once
# -----------------------------
VECTORSTORE: Optional[PGVector] = None


def startup_ingest() -> None:
    global VECTORSTORE
    ensure_pgvector_extension(DATABASE_URL)

    docs = load_all_pdfs(DOCS_DIR)
    chunks = chunk_documents(docs)

    VECTORSTORE = build_vectorstore(chunks)
    logger.info(
        "Ingested %s pages, %s chunks into '%s'.",
        len(docs),
        len(chunks),
        COLLECTION_NAME,
    )


# -----------------------------
# Flask endpoints
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "collection": COLLECTION_NAME})


@app.route("/qa", methods=["POST"])
def qa():
    global VECTORSTORE
    try:
        if VECTORSTORE is None:
            return jsonify({"error": "Vector store not initialized."}), 500

        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing 'question' in JSON body."}), 400

        result = answer_with_rag(VECTORSTORE, question, k=4)
        return jsonify(result)

    except Exception as e:
        logger.error("Internal error: %s", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    if INGEST_ON_STARTUP:
        startup_ingest()
    else:
        logger.info("INGEST_ON_STARTUP=false -> Skipping ingestion.")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
