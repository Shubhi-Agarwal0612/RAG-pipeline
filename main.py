from fastapi import FastAPI, UploadFile, File, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
from groq import Groq
import pdfplumber
import psycopg2
import os
import shutil
import re
from collections import Counter
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    conn = psycopg2.connect(
        dbname="rag_app",
        user="postgres",
        password="postgres",
        host="localhost"
    )
    register_vector(conn)
    return conn


# FUNCTION 1: Process Document



def process_document(file_path):
    pdf = pdfplumber.open(file_path)

    # PHASE 1: Extract raw text from every page (keep \n intact)
    raw_pages = []
    for page_no, page in enumerate(pdf.pages):
        text = page.extract_text()
        if not text:
            continue
        raw_pages.append({"page_no": page_no + 1, "text": text})
    pdf.close()

    # PHASE 2: Detect and remove headers, footers, page numbers
    all_lines = []
    for page in raw_pages:
        lines = page["text"].split('\n')
        all_lines.extend(set(lines))

    line_counts = Counter(all_lines)
    threshold = len(raw_pages) * 0.5
    headers_footers = {
        line for line, count in line_counts.items()
        if count > threshold and len(line.strip()) > 0
    }

    for page in raw_pages:
        lines = page["text"].split('\n')
        cleaned_lines = [line for line in lines if line not in headers_footers]
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'Page\s*\d+(\s*(of|/)\s*\d+)?', '', cleaned_text)
        page["text"] = cleaned_text

    # PHASE 3: Clean whitespace and chunk with overlap
    final_list = []
    for page in raw_pages:
        text = ' '.join(page["text"].split())
        if not text.strip():
            continue
        chunk_index = 0
        for i in range(0, len(text), 400):
            chunk = text[i : i + 500]
            final_list.append({
                "page_no": page["page_no"],
                "chunk_index": chunk_index,
                "text": chunk
            })
            chunk_index += 1

    # PHASE 4: TF-IDF keyword extraction
    if final_list:
        texts = [chunk["text"] for chunk in final_list]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        for i, chunk in enumerate(final_list):
            scores = tfidf_matrix[i].toarray()[0]
            top_indices = scores.argsort()[-6:]
            keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
            chunk["keywords"] = keywords

    return final_list


# FUNCTION 2: Embed and Store
def embed_and_store(document_id, final_list):
    conn = get_db()
    cur = conn.cursor()
    for chunk in final_list:
        vector = model.encode(chunk["text"] + " " + ", ".join(chunk["keywords"])).tolist()
        chunk_id = f"doc{document_id}_p{chunk['page_no']}_c{chunk['chunk_index']}"
        metadata = json.dumps({
            "page_no": chunk["page_no"],
            "chunk_index": chunk["chunk_index"],
            "keywords": chunk["keywords"]
        })
        cur.execute(
            "INSERT INTO vector_embeddings (document_id, chunk_id, text, metadata, vector) VALUES (%s, %s, %s, %s, %s)",
            (document_id, chunk_id, chunk["text"], metadata, vector),
        )
    conn.commit()
    cur.close()
    conn.close()
    
#Function 3 :  used to convert a follow-up question into a standalone question, by providing the conversation history as context to the model. This is important for accurate retrieval and evaluation, as it ensures the question is fully self-contained and can be understood without any external reference. 
rewrite_system_prompt = """You rewrite the user's most recent question into a fully standalone question, using the earlier conversation only when needed.

Only rewrite if the latest question literally cannot be understood on its own — for example it uses pronouns like "it", "its", or "they" that refer to something in the earlier conversation, or starts with words like "and" or "but", or is otherwise incomplete. In that case, use the earlier conversation to rewrite it into a complete, self-contained question.

If the latest question already makes sense on its own, or introduces a new topic — even one unrelated to the earlier conversation — return it completely unchanged, word for word. Do not try to connect an unrelated question to the earlier conversation.

Never answer the question. Your output must always itself be a question — only the rewritten (or unchanged) question, with no preamble, explanation, or answer."""


def rewrite(convo_history, question):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": rewrite_system_prompt},
            *convo_history,
            {"role": "user", "content": question},
        ],
    )
    standalone_question = response.choices[0].message.content
    return standalone_question


# FUNCTION 4: Query
# ── At the top of the file, with your other model load ──
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def query(question, document_ids):
    vector = model.encode(question).tolist()
    chunks = []
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT text, chunk_id, document_id, 1 - (vector <=> %s::vector) as similarity
           FROM vector_embeddings
           WHERE document_id = ANY(%s)
           ORDER BY similarity DESC
           LIMIT 20""",
        (vector, document_ids),
    )
    relevant_chunks = cur.fetchall()
    cur.close()
    conn.close()

    if not relevant_chunks:
        answer = "I don't have relevant information in the selected documents to answer this question."
        return answer, chunks

    # Build the candidate chunk dicts from the 20 retrieved rows
    for row in relevant_chunks:
        chunk_dict = {}
        chunk_dict["chunk_id"] = row[1]
        chunk_dict["text"] = row[0]
        chunk_dict["similarity_score"] = row[3]
        chunks.append(chunk_dict)

    # ── Rerank: score each (question, chunk) pair, keep the best 5 ──
    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = reranker.predict(pairs)
    final_list = list(zip(chunks, scores))
    final_list.sort(key=lambda x: x[1], reverse=True)
    top_5_pairs = final_list[:5]
    final_chunks = []
    for i in top_5_pairs:
        final_chunks.append(i[0])          # i[0] = chunk dict, i[1] = score

    # Build the sources string from the reranked top 5
    chunks_string = ""
    for i, row in enumerate(final_chunks):
        chunks_string += f"Source {i + 1} [{row['chunk_id']}]: {row['text']}\n\n"

    system_prompt = """You are an exceptional research assistant who answers questions from provided documents.
Only answer from the provided context, do not add any extra information from training knowledge.
In case there is conflicting information, acknowledge the conflict and provide both sides, do not pick a side.
If the provided context does not contain enough information, say 'I don't have the information to answer this.'
If the context is only partially relevant, answer what you can and state what you cannot answer.
When using information, reference which source it came from."""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{chunks_string}"},
        ],
    )

    answer = response.choices[0].message.content
    return answer, final_chunks


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(file_path)
    file_type = file.filename.split(".")[-1]

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO documents (file_name, file_size, file_type, file_path, status)
           VALUES (%s, %s, %s, %s, %s)
           RETURNING document_id""",
        (file.filename, file_size, file_type, file_path, "uploaded"),
    )
    document_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    return {"document_id": document_id, "file_name": file.filename, "status": "uploaded"}


@app.post("/embed/{document_id}")
def embed_document(document_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT file_path, status FROM documents WHERE document_id = %s", (document_id,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = row[0]

    cur.execute("UPDATE documents SET status = 'processing' WHERE document_id = %s", (document_id,))
    conn.commit()
    cur.close()
    conn.close()

    try:
        chunks = process_document(file_path)
        embed_and_store(document_id, chunks)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE documents SET status = 'ready' WHERE document_id = %s", (document_id,))
        conn.commit()
        cur.close()
        conn.close()

        return {"document_id": document_id, "status": "ready", "chunks_created": len(chunks)}

    except Exception as e:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE documents SET status = 'failed' WHERE document_id = %s", (document_id,))
        conn.commit()
        cur.close()
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """SELECT document_id, file_name, file_size, file_type, status, date_uploaded
           FROM documents ORDER BY date_uploaded DESC"""
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    documents = []
    for row in rows:
        documents.append({
            "document_id": row[0],
            "file_name": row[1],
            "file_size": row[2],
            "file_type": row[3],
            "status": row[4],
            "date_uploaded": str(row[5]),
        })

    return {"documents": documents}


class QueryRequest(BaseModel):
    question: str
    document_ids: list[int]
    history: list[dict] = []        # NEW: browser sends conversation so far (defaults to empty)


@app.post("/query")
def query_documents(request: QueryRequest):
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="Select at least one document")

    # Rewrite the follow-up into a standalone question using history
    standalone_question = rewrite(request.history, request.question)

    # query() returns (answer, chunks) — unpack both
    answer, chunks = query(standalone_question, request.document_ids)

    return {
        "answer": answer,
        "question": request.question,
        "standalone_question": standalone_question,   # so you can SEE the rewrite in the demo
    }


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
