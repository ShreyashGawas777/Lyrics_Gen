"""
rag_engine.py
-------------
Core Retrieval-Augmented Generation (RAG) engine for lyric generation.
"""

import chromadb
from chromadb.utils import embedding_functions
import subprocess
import shlex

# 1️⃣ Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="../db/chroma")
collection = client.get_collection("lyrics")

# 2️⃣ Set up the embedding function (Ollama)
embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")

def retrieve_lyrics(query, top_k=3):
    """
    Retrieve the most similar lyrics from ChromaDB.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    retrieved = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        retrieved.append(f"{meta['title']} ({meta['artist']}): {doc}")
    
    return retrieved

def generate_lyrics(prompt, retrieved_lyrics, model_name="phi3:mini"):
    """
    Generates new lyrics using Ollama LLM.
    Combines user prompt with retrieved lyrics as context.
    """
    context = "\n".join(retrieved_lyrics)
    full_prompt = f"Based on the following lyrics:\n{context}\n\nGenerate a new song about: {prompt}"
    
    cmd = f"ollama run {model_name} {shlex.quote(full_prompt)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    return result.stdout.strip()

# 3️⃣ Test example
if __name__ == "__main__":
    theme = "midnight city lights and loneliness"
    retrieved = retrieve_lyrics(theme)
    print("🎵 Retrieved Lyrics:")
    print("\n".join(retrieved), "\n")
    
    new_lyrics = generate_lyrics(theme, retrieved)
    print("✨ Generated Lyrics:")
    print(new_lyrics)

