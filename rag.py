# rag.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from anthropic import Anthropic
import hnswlib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
from tqdm import tqdm
import pickle

# Initialize Claude, tokenizer, and model
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Paths for saving/loading state
INDEX_PATH = "blacks_law_index.bin"
CHUNKS_PATH = "blacks_law_chunks.pkl"

# Initialize HNSW index object
dim = 768
index = hnswlib.Index(space="cosine", dim=dim)
chunks = []

def init_index():
    global index, chunks
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index.load_index(INDEX_PATH, max_elements=50000)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from saved state.")
    else:
        index.init_index(max_elements=20000, ef_construction=200, M=32)
        chunks = []
        print("Initialized new index.")
    index.set_ef(1000)

init_index()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def index_document(file_path):
    global index, chunks
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        text = "".join(page.extract_text() for page in pdf.pages)
    
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if len(chunks) > 20000:
        print(f"Warning: PDF has {len(chunks)} chunks, exceeding capacity (20000). Truncating.")
        chunks = chunks[:20000]
    
    batch_size = 100
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i + batch_size]
        batch_embeddings = np.array([get_embedding(chunk) for chunk in batch])
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=20000, ef_construction=200, M=32)
    index.add_items(embeddings, list(range(len(chunks))))
    index.set_ef(1000)
    
    index.save_index(INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Indexed and saved {len(chunks)} chunks from {file_path}")


def query_rag(question):
    if not chunks or index.get_current_count() == 0:
        return "No document indexed yet. Please upload a file first."
    
    query_embedding = get_embedding(question)
    query_embedding = np.expand_dims(query_embedding, axis=0)
    
    labels, distances = index.knn_query(query_embedding, k=3)
    context = "\n".join(chunks[idx] for idx in labels[0])
    
    messages = [
        {
            "role": "user",
            "content": (
                f"Context from Black's Law Dictionary:\n{context}\n\n"
                f"Question: {question}\n"
                f"Provide a concise, accurate answer based on the context."
            )
        }
    ]
    
    response = claude.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        system="You are an expert in legal terminology from Black's Law Dictionary.",
        messages=messages
    )
    return response.content[0].text