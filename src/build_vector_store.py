# src/build_vector_store.py

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import os

# --- 1. Configuration ---
DATA_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    """
    Main function to build and persist the vector store.
    """
    print("--- Starting Vector Store Creation ---")

    # --- 2. Load Data ---
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure required columns exist
    if 'narrative' not in df.columns or 'product' not in df.columns or 'complaint_id' not in df.columns:
        raise ValueError("CSV must contain 'narrative', 'product', and 'complaint_id' columns.")
        
    loader = DataFrameLoader(df, page_content_column="narrative")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # --- 3. Text Chunking ---
    print(f"Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # --- 4. Embedding Model ---
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    # This model runs locally and is great for general-purpose text embeddings.
    model_kwargs = {'device': 'cpu'} # Use 'cuda' for GPU
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")

    # --- 5. Create and Persist Vector Store ---
    print(f"Creating and persisting vector store at {VECTOR_STORE_PATH}...")
    # Using Chroma's from_documents to handle embedding and storage in one step.
    # This will take some time as it embeds all chunks.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    print("--- Vector Store Creation Complete ---")
    print(f"Total vectors in store: {vector_store._collection.count()}")


if __name__ == '__main__':
    main()