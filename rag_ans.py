import faiss
import pickle
import numpy as np
import requests
import os
import re
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")

try:
    index = faiss.read_index("faiss_index.bin")
    with open("text_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    chunk_texts = [chunk.page_content for chunk in chunks]
except FileNotFoundError:
    print("Error: faiss_index.bin or text_chunks.pkl not found.")
    print("Please run embed_chunks.py first to create these files.")
    exit()
    
def get_query_embedding(question):
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    payload = {"input": [question], "model": EMBED_MODEL, "input_type": "query"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Embedding API error: {e}")
        return None

def expand_query_with_llm(question):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json", 
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    
    expansion_prompt = f"""You are an expert at converting a user's question into effective search queries for a vector database.
Based on the user's question, generate 3 to 5 specific, diverse questions that would help retrieve the most relevant and comprehensive information.
Return the questions as a numbered list. Do not add any other commentary.

User Question: "{question}"
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": expansion_prompt}],
        "max_tokens": 200, "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        expanded_queries = [line.split('. ', 1)[1] for line in content.strip().split('\n') if '. ' in line]
        if question not in expanded_queries:
            expanded_queries.insert(0, question)
        return expanded_queries
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        print(f"Warning: Could not expand query, using original question. Error: {e}. Body: {body}")
        return [question]

def ask_llm(context, question):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json", 
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    
    synthesis_prompt = f"""You are a helpful AI assistant. Answer the user's question based *only* on the following context.
If the context does not contain the answer, state that you cannot answer with the information provided.
Be comprehensive and synthesize information from all parts of the context.

Context:
{context}

Question: {question}
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": synthesis_prompt}],
        "max_tokens": 4096, "temperature": 0.3, "top_p": 0.9
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        print(f"LLM API error: {e}. Body: {body}")
        return None


def ask_llm_open(question):
    """Ask the LLM without context (general fallback)."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ],
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        body = e.response.text if hasattr(e, "response") and e.response is not None else "<no body>"
        print(f"Open LLM API error: {e}. Body: {body}")
        return None


def _looks_like_refusal(text: str) -> bool:
    if not text:
        return True
    t = text.lower()
    phrases = [
        "cannot answer",
        "can't answer",
        "not contain the answer",
        "not in the context",
        "information provided",
        "insufficient information",
        "do not have enough information",
    ]
    return any(p in t for p in phrases)

if __name__ == "__main__":
    question = input("Enter your question: ")
    
    print("\nExpanding query...")
    expanded_queries = expand_query_with_llm(question)
    print(f"-> Generated queries: {expanded_queries}")
    print("\nRetrieving relevant chunks...")
    retrieved_indices = set()
    for q in expanded_queries:
        query_emb = get_query_embedding(q)
        if query_emb:
            D, I = index.search(np.array([query_emb]).astype("float32"), k=2)
            for i in I[0]:
                retrieved_indices.add(i)
    
    if not retrieved_indices:
        print("No relevant chunks found. Answering from general knowledge...\n")
        answer = ask_llm_open(question)
        if answer:
            print("Answer:\n")
            print(answer)
        else:
            print("Could not generate an answer.")
    else:
        top_chunks = [chunk_texts[i] for i in retrieved_indices]
        context = "\n\n---\n\n".join(top_chunks)

        print("\nSynthesizing answer...")
        answer = ask_llm(context, question)
        if not answer or _looks_like_refusal(answer):
            print("The document didn't contain the answer. Answering from general knowledge...\n")
            answer = ask_llm_open(question)
        if answer:
            print("\nAnswer from LLM:\n")
            print(answer)
        else:
            print("Could not generate an answer.")