import requests
import faiss
import numpy as np
import pickle
from load_chunk import load_chunk
import os 

print("Loading and splitting chunks from data.txt...")
chunks = load_chunk("data.txt")
texts = [chunk.page_content for chunk in chunks]

def get_embedding(texts):
    """Gets embeddings for a list of text passages."""
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"
    }
    
    payload = {
        "input": texts,
        "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "input_type": "passage"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
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