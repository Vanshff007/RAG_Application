# RAG 

A Retrieval-Augmented Generation (RAG) system that allows users to ask questions about uploaded documents or a default knowledge base. The system uses NVIDIA's embedding and LLM APIs to retrieve relevant information and generate comprehensive answers.

## Features

- **Multiple File Format Support**: Upload and process PDF, TXT, or DOCX files
- **Smart Query Expansion**: Automatically generates multiple search queries to improve retrieval accuracy
- **Hybrid Answering Mode**: Falls back to general knowledge when document context is insufficient
- **Document-Only Mode**: Option to restrict answers strictly to uploaded document content
- **Vector Search**: Uses FAISS for efficient similarity search across document chunks
- **Interactive Web UI**: Built with Streamlit for easy interaction
- **Command-Line Interface**: Alternative CLI tool for batch processing

## Architecture

The system implements a classic RAG pipeline:

1. **Document Processing**: Splits documents into chunks using recursive character text splitting
2. **Embedding Generation**: Converts text chunks into vector embeddings using NVIDIA's embedding model
3. **Vector Storage**: Stores embeddings in a FAISS index for fast retrieval
4. **Query Expansion**: Uses LLM to generate related queries for better recall
5. **Retrieval**: Searches for the most relevant chunks based on query similarity
6. **Answer Generation**: Synthesizes a comprehensive answer using retrieved context

## Prerequisites

- Python 3.8+
- NVIDIA API Key (get it from [NVIDIA AI Playground](https://build.nvidia.com))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DK2325/RAG_Application
cd RAG_Application
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root:
```env
NVIDIA_API_KEY=your_api_key_here
EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
LLM_MODEL=meta/llama-3.3-70b-instruct
```

## Usage

### Web Interface (Streamlit)

Launch the interactive web application:

```bash
streamlit run app.py
```

Then:
1. Upload a document (PDF, TXT, or DOCX) or use the default `data.txt`
2. Enter your question in the text input
3. Optionally check "Use only the uploaded document" to restrict answers to document content
4. Click "Generate Answer"

### Command-Line Interface

For pre-indexed documents:

1. First, create embeddings from your document:
```bash
python embed_chunks.py
```

This generates `faiss_index.bin` and `text_chunks.pkl` files.

2. Then run queries:
```bash
python rag_ans.py
```

Enter your question when prompted.

### Processing Custom Documents

To process a different document with the CLI:

1. Update the file path in `load_chunk.py` (line 14)
2. Run `embed_chunks.py` to generate new embeddings
3. Use `rag_ans.py` to ask questions

## Project Structure

```
.
├── app.py                 # Streamlit web application
├── rag_ans.py            # Command-line interface
├── embed_chunks.py       # Embedding generation script
├── load_chunk.py         # Document loading and chunking utilities
├── data.txt              # Default knowledge base
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
├── faiss_index.bin      # Generated FAISS index (ignored in git)
└── text_chunks.pkl      # Generated chunks (ignored in git)
```

### Retrieval Parameters

Modify the number of chunks retrieved per query in `app.py` or `rag_ans.py`:
```python
index.search(query_emb, k=2)  # Change k to retrieve more/fewer chunks
```

### Model Selection

Change models in your `.env` file:
```env
EMBED_MODEL=nvidia/llama-3.2-nv-embedqa-1b-v2
LLM_MODEL=meta/llama-3.3-70b-instruct
```

## How It Works

### Query Expansion
The system uses an LLM to generate 3-5 related questions for each user query, improving retrieval coverage:
- Original: "What is RAG?"
- Expanded: "What does RAG stand for?", "How does RAG work?", "What are the components of RAG?"

### Fallback Mechanism
When the document doesn't contain relevant information:
- **Default mode**: Falls back to general LLM knowledge
- **Document-only mode**: Informs user that the answer isn't in the document

### Refusal Detection
The system detects when the LLM indicates it cannot answer from context and automatically triggers fallback behavior.

## API Rate Limits

Be aware of NVIDIA API rate limits. The system includes:
- Timeout configurations (20-60 seconds)
- Error handling with detailed messages
- Sequential processing to avoid overwhelming the API

## Troubleshooting

### "Embedding API error"
- Check your NVIDIA API key is valid
- Ensure you have API credits remaining
- Verify your internet connection

### "faiss_index.bin not found"
- Run `embed_chunks.py` first to generate the index
- Ensure the file is in the same directory

### Poor Answer Quality
- Try adjusting chunk size/overlap
- Increase the number of retrieved chunks (k parameter)
- Use query expansion (enabled by default)
- Upload a more comprehensive document
- 
## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Embeddings and LLM inference via [NVIDIA AI Endpoints](https://build.nvidia.com)
- UI built with [Streamlit](https://streamlit.io/)
