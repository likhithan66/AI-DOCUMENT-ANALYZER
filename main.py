
from io import BytesIO

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from app.chunking import split_text


# --------------------------------------------------
# 1. Create FastAPI application
# --------------------------------------------------

app = FastAPI(title="Document Analyzer AI")


# --------------------------------------------------
# 2. Configure CORS for React frontend
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 3. Temporary storage for uploaded PDF chunks
# --------------------------------------------------

document_chunks = []
document_filename = None


# --------------------------------------------------
# 4. Request models using Pydantic
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class DocumentQuestion(BaseModel):
    question: str
    context: str


# --------------------------------------------------
# 5. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Document Analyzer AI is running!"
    }


# --------------------------------------------------
# 6. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# 7. Upload PDF, extract text, and create chunks
# --------------------------------------------------

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global document_chunks, document_filename

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        reader = PdfReader(BytesIO(file_bytes))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported."
            )

        extracted_text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read this PDF."
        )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text found. This PDF may be scanned "
                "and require OCR."
            )
        )

    chunks = split_text(
        extracted_text,
        chunk_size=500,
        overlap=50
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="Could not create chunks from this PDF."
        )

    # Save the latest uploaded document in memory
    document_chunks = chunks
    document_filename = file.filename

    return {
        "filename": document_filename,
        "total_pages": len(reader.pages),
        "total_chunks": len(document_chunks),
        "chunks": document_chunks
    }


# --------------------------------------------------
# 8. Check Ollama connection
# --------------------------------------------------

@app.get("/ollama-status")
def ollama_status():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )
        response.raise_for_status()

        return {
            "status": "connected",
            "models": [
                model["name"]
                for model in response.json().get("models", [])
            ]
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Ollama."
        )


# --------------------------------------------------
# 9. Ask Ollama a general question
# --------------------------------------------------

@app.post("/ollama-test")
def ollama_test(request: ChatRequest):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:latest",
                "messages": [
                    {
                        "role": "user",
                        "content": request.question
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return {
            "question": request.question,
            "answer": data["message"]["content"]
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Ollama request failed. Check that Ollama is running."
        )


# --------------------------------------------------
# 10. Ask Ollama using manually supplied context
# --------------------------------------------------

@app.post("/ask-document")
def ask_document(request: DocumentQuestion):
    prompt = f"""
Answer the question using only the document context below.

If the answer is not present in the context, say:
"The answer was not found in the provided document."

DOCUMENT CONTEXT:
{request.context}

QUESTION:
{request.question}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:latest",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return {
            "question": request.question,
            "answer": data["message"]["content"]
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Ollama request failed. Check that Ollama is running."
        )


# --------------------------------------------------
# 11. Retrieve relevant chunks using keyword matching
#     and answer using Ollama
# --------------------------------------------------

@app.post("/ask-uploaded-document")
def ask_uploaded_document(request: ChatRequest):
    if not document_chunks:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    question_words = {
        word.lower().strip(".,?!:;()[]{}\"'")
        for word in request.question.split()
        if len(word.strip(".,?!:;()[]{}\"'")) > 2
    }

    if not question_words:
        raise HTTPException(
            status_code=400,
            detail="Please enter a meaningful question."
        )

    scored_chunks = []

    for chunk in document_chunks:
        chunk_words = {
            word.lower().strip(".,?!:;()[]{}\"'")
            for word in chunk.split()
        }

        score = len(question_words & chunk_words)

        if score > 0:
            scored_chunks.append((score, chunk))

    # Sort chunks by keyword match score
    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    relevant_chunks = [
        chunk
        for score, chunk in scored_chunks[:3]
    ]

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant text found for this question."
        )

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the supplied document context.
Do not invent facts or use outside knowledge.
If the answer is not present in the context, clearly say
that the answer was not found in the provided document.

DOCUMENT:
{document_filename}

DOCUMENT CONTEXT:
{context}

QUESTION:
{request.question}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:latest",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        data = response.json()

        return {
            "filename": document_filename,
            "question": request.question,
            "retrieved_chunks": relevant_chunks,
            "answer": data["message"]["content"]
        }

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Ollama request failed. Check that Ollama is running."
        )
