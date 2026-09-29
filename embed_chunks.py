import requests
import faiss
import numpy as np
import pickle
from load_chunk import load_chunk
from local_embed import local_embed
import os 

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
EMBED_BATCH_SIZE = 100

print("Loading and splitting chunks from data.txt...")
chunks = load_chunk("data.txt")
texts = [chunk.page_content for chunk in chunks]

def get_embedding(texts):
    """Gets embeddings for a list of text passages."""
    if EMBED_PROVIDER == "local":
        return local_embed(texts, "passage")
    url = f"{LLM_BASE_URL}/embeddings"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    
    embeddings = []
    try:
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            payload = {"input": texts[start:start + EMBED_BATCH_SIZE], "model": EMBED_MODEL}
            # input_type is an NVIDIA-only field; other providers reject it
            if "nvidia.com" in LLM_BASE_URL:
                payload["input_type"] = "passage"
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            embeddings.extend(item["embedding"] for item in response.json()["data"])
        return embeddings
    except requests.exceptions.RequestException as e:
        print(f"API ERROR: {e}")
        return []

print(f"Generating embeddings for {len(texts)} chunks...")
embeddings = get_embedding(texts)

if embeddings:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, "faiss_index.bin")

    with open("text_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Successfully saved {len(embeddings)} embeddings to faiss_index.bin and chunks to text_chunks.pkl")
else:
    print("No embeddings were generated. Exiting.")