import json
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
import os

# Make sure the db folder exists
os.makedirs("../db/chroma", exist_ok=True)

# Initialize ChromaDB client (persistent local store)
client = chromadb.PersistentClient(path="../db/chroma")

# Embedding function (Ollama embedding)
embedding_fn = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Create or get collection
collection = client.get_or_create_collection(
    name="lyrics",
    embedding_function=embedding_fn
)


def build_db_from_json(json_path):
    """
    Reads a JSON file of songs and stores their embeddings in ChromaDB.
    Expected format:
    [
        {"title": "Song A", "artist": "Artist X", "lyrics": "lyrics text..."},
        {"title": "Song B", "artist": "Artist X", "lyrics": "lyrics text..."}
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        songs = json.load(f)

    print(f" Found {len(songs)} songs. Embedding and storing...")
    for i, song in tqdm(enumerate(songs), total=len(songs)):
        collection.add(
            documents=[song["lyrics"]],
            metadatas=[{"title": song["title"], "artist": song["artist"]}],
            ids=[f"song_{i}"]
        )
    print(" Lyrics embedded and stored in ChromaDB successfully.")

if __name__ == "__main__":
    # Example JSON file — replace this with your artist data
    build_db_from_json("../data/artist_x_lyrics.json")
