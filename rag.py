import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv  # Add this import
from anthropic import Anthropic
import hnswlib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pdfplumber
from tqdm import tqdm
import pickle


# Initialize Claude, tokenizer, and model
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY is not set in the .env file or environment. Please set it before running the script.")
claude = Anthropic(api_key=api_key)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Paths for saving/loading state
INDEX_PATH = "./blacks_law_index.bin"
CHUNKS_PATH = "./blacks_law_chunks.pkl"
METADATA_PATH = "./blacks_law_metadata.pkl"

# Initialize HNSW index
dim = 768
index = hnswlib.Index(space="cosine", dim=dim)
chunks = []
metadata = []

def init_index():
    global index, chunks, metadata
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(METADATA_PATH):
        index.load_index(INDEX_PATH, max_elements=50000)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from saved state.")
    else:
        index.init_index(max_elements=50000, ef_construction=200, M=16)
        chunks = []
        metadata = []
        print("Initialized new index.")
    index.set_ef(max(10, min(100, len(chunks) * 2)))

init_index()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def index_document(file_path):
    global index, chunks, metadata
    chunks = []
    metadata = []

    with pdfplumber.open(file_path) as pdf:
        for page in tqdm(pdf.pages, desc="Extracting pages"):
            # Split page into two columns
            width = page.width
            left_column = page.crop((0, 0, width / 2, page.height)).extract_text() or ""
            right_column = page.crop((width / 2, 0, width, page.height)).extract_text() or ""
            print(f"Left column:\n{left_column}")
            print(f"Right column:\n{right_column}")

            # Process each column
            for column_text in [left_column, right_column]:
                if not column_text.strip():
                    continue
                # Split into definitions: all-caps term followed by definition until next all-caps term
                pattern = r"([A-Z][A-Z\s\-\']{2,})\.?\s*(.*?)(?=\n[A-Z][A-Z\s\-\']{2,}\.|$)"
                entries = re.findall(pattern, column_text, re.DOTALL)
                for term, definition in entries:
                    term = term.strip()
                    definition = definition.strip()
                    if term and definition:
                        chunks.append(f"{term}: {definition}")
                        metadata.append(term)
                print(f"Column entries: {entries}")

    print(f"Extracted chunks: {chunks}")
    print(f"Metadata: {metadata}")

    if not chunks:
        print("No valid definitions found in the PDF.")
        return

    if len(chunks) > 50000:
        print(f"Warning: PDF has {len(chunks)} chunks, exceeding capacity (50000). Truncating.")
        chunks = chunks[:50000]
        metadata = metadata[:50000]

    batch_size = 100
    embeddings = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[i:i + batch_size]
        batch_embeddings = np.array([get_embedding(chunk) for chunk in batch])
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=50000, ef_construction=200, M=16)
    index.add_items(embeddings, list(range(len(chunks))))
    index.set_ef(max(10, min(100, len(chunks) * 2)))

    index.save_index(INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Indexed and saved {len(chunks)} chunks from {file_path}")

def query_rag(question):
    if not chunks or index.get_current_count() == 0:
        return "No document indexed yet. Please upload a file first."

    query_embedding = get_embedding(question)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    k = min(3, index.get_current_count())
    labels, distances = index.knn_query(query_embedding, k=k)
    context = []
    for idx in labels[0]:
        term = metadata[idx]
        if term.lower() in question.lower():
            context.insert(0, chunks[idx])
        else:
            context.append(chunks[idx])

    context_text = "\n".join(context[:3])
    messages = [
        {
            "role": "user",
            "content": (
                f"Context from Black's Law Dictionary:\n{context_text}\n\n"
                f"Question: {question}\n"
                f"Provide a concise, accurate answer based on the context."
            )
        }
    ]

    response = claude.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        system="You are an expert in legal terminology from Black's Law Dictionary and a professional in common law and well read in case law in US",
        messages=messages
    )
    return response.content[0].text