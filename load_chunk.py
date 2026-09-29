from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunk(file_path: str):
    """Loads a text file and splits it into chunks."""
    with open(file_path, encoding="utf-8") as f:
        raw_docs = [Document(page_content=f.read(), metadata={"source": file_path})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(raw_docs)
    return chunks

if __name__ == "__main__":
    chunks = load_chunk("data.txt") 
    print(f"Loaded and split into {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---\n{chunk.page_content}")