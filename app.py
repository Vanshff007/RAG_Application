import streamlit as st
import requests
import numpy as np
import faiss
from PyPDF2 import PdfReader
from docx import Document
from io import StringIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re
from local_embed import local_embed

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# "local" embeds on this machine (see local_embed.py); anything else calls LLM_BASE_URL
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "api")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBED_BATCH_SIZE = 100
# Tasks like "summarize" send the whole document to the LLM; cap it so large
# files stay inside free-tier token limits (about 4 characters per token)
MAX_TASK_CHARS = int(os.getenv("MAX_TASK_CHARS", "20000"))
NOT_FOUND = "Not mentioned in the document"


def embed_payload(texts, input_type):
    payload = {"input": texts, "model": EMBED_MODEL}
    # input_type is an NVIDIA-only field; other providers reject it
    if "nvidia.com" in LLM_BASE_URL:
        payload["input_type"] = input_type
    return payload


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_txt(file):
    return StringIO(file.getvalue().decode("utf-8")).read()

def read_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def read_data_txt():
    with open("data.txt", encoding="utf-8") as f:
        return f.read()


def is_task_question(question: str) -> bool:
    q = question.lower().strip()
    task_verbs = [
        "summarize", "summary", "explain", "rewrite", "rephrase",
        "extract", "list", "analyze", "analyse", "compare",
        "format", "convert", "simplify", "improve",
        "generate", "create", "outline", "describe"
    ]
    return any(v in q for v in task_verbs)


def embed_chunks(chunks):
    if EMBED_PROVIDER == "local":
        return local_embed(chunks, "passage")
    url = f"{LLM_BASE_URL}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    embeddings = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        payload = embed_payload(chunks[start:start + EMBED_BATCH_SIZE], "passage")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        embeddings.extend(item["embedding"] for item in response.json()["data"])
    return embeddings

def get_query_embedding(question):
    if EMBED_PROVIDER == "local":
        return local_embed([question], "query")[0]
    url = f"{LLM_BASE_URL}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    payload = embed_payload([question], "query")
    response = requests.post(url, json=payload, headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def chat(prompt, max_tokens=2048):
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def run_task_on_document(document_text, task):
    prompt = f"""
You are an AI assistant working ONLY with the given document.

Perform the user's task using the document content.
Do NOT say that information is missing.
Do NOT refuse.
Always attempt the task using what is available.

Document:
{document_text}

Task:
{task}
"""
    return chat(prompt)

def ask_llm_fact(context, question):
    prompt = f"""
Answer the question using ONLY the context below.
If the answer truly does not exist, say "{NOT_FOUND}".

Context:
{context}

Question:
{question}
"""
    return chat(prompt)

def ask_llm_open(question):
    return chat(question)


@st.cache_resource(max_entries=5, show_spinner=False)
def build_index(document_text):
    """Splits and embeds a document once; later questions on it reuse the index."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(document_text)
    embeddings = embed_chunks(texts)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, texts


def api_error_message(error):
    if error.response is None:
        return f"Could not reach the API: {error}"
    try:
        detail = error.response.json()
        if isinstance(detail, list):
            detail = detail[0]
        detail = detail["error"]["message"]
    except Exception:
        detail = error.response.text[:300]
    return f"API error {error.response.status_code}: {detail}"


st.title("RAG")

uploaded_file = st.file_uploader(
    "Upload PDF / TXT / DOCX (optional)",
    type=["pdf", "txt", "docx"]
)

question = st.text_input("Ask a question or give a task:")
use_doc_only = st.checkbox(
    "Use only the document",
    value=True,
    help="When off, questions the document can't answer are answered from general knowledge."
)

if st.button("Generate Answer") and question.strip():

    with st.spinner("Reading document..."):
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                document_text = read_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                document_text = read_txt(uploaded_file)
            else:
                document_text = read_docx(uploaded_file)
        else:
            document_text = read_data_txt()

    if not document_text.strip():
        st.error("No text found in the document. Scanned PDFs (images of pages) are not supported.")
        st.stop()

    try:
        if is_task_question(question):
            if len(document_text) > MAX_TASK_CHARS:
                st.warning(
                    f"Document is long, so only the first {MAX_TASK_CHARS:,} of "
                    f"{len(document_text):,} characters were used for this task."
                )
            with st.spinner("Performing task on document..."):
                result = run_task_on_document(document_text[:MAX_TASK_CHARS], question)
            st.success("Result")
            st.write(result)
            st.stop()

        with st.spinner("Embedding document..."):
            index, texts = build_index(document_text)

        with st.spinner("Searching document..."):
            q_emb = get_query_embedding(question)
            D, I = index.search(np.array([q_emb]).astype("float32"), k=min(5, len(texts)))
            context = "\n\n---\n\n".join(texts[i] for i in I[0])

        with st.spinner("Generating answer..."):
            answer = ask_llm_fact(context, question)
            if not use_doc_only and NOT_FOUND.lower() in answer.lower():
                st.info("The document doesn't cover this. Answering from general knowledge.")
                answer = ask_llm_open(question)
        st.success("Answer")
        st.write(answer)
    except requests.exceptions.RequestException as e:
        st.error(api_error_message(e))
