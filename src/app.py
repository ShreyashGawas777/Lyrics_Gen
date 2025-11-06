import os
import json
import streamlit as st
import chromadb
import ollama  # Import the official Ollama python package
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
# Get the directory of the current script (src)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one level up to the project root (rag_lyrics_gen)
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "db", "chroma")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_DEFAULT = "phi3:mini"  # Your installed model

# -----------------------------
# BUILD / CONNECT TO DB
# -----------------------------
@st.cache_resource
def get_chroma_client():
    """Get a persistent ChromaDB client."""
    # Ensure the DB path exists
    os.makedirs(DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=DB_PATH)

def get_chroma_collection(name="lyrics"):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    try:
        collection = client.get_collection(name)
    except Exception:
        st.info(f"Creating new collection: {name}")
        collection = client.create_collection(name)
    return collection

def build_db_from_json(file_path):
    """Reads uploaded JSON and embeds lyrics into Chroma."""
    from chromadb.utils import embedding_functions
    
    st.write(" 📦 Building database from uploaded JSON…")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not data:
        st.error(" ❌ JSON file is empty or invalid.")
        return

    # Use Ollama for embeddings
    # Make sure 'nomic-embed-text' is pulled: ollama pull nomic-embed-text
    try:
        embedding_function = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
    except Exception as e:
        st.error(f"Error initializing embedding model. Is 'nomic-embed-text' pulled? Error: {e}")
        return

    client = get_chroma_client()

    # Clear old collection if it exists, to rebuild
    try:
        client.delete_collection("lyrics")
    except Exception:
        pass # Collection didn't exist, which is fine

    collection = client.create_collection("lyrics", embedding_function=embedding_function)
    
    docs, metas, ids = [], [], []
    for i, song in enumerate(tqdm(data, desc="Embedding songs")):
        docs.append(song["lyrics"])
        metas.append({"title": song.get("title", f"song_{i}"), "artist": song.get("artist", "Unknown")})
        ids.append(f"song_{i}") 
    
    collection.add(documents=docs, metadatas=metas, ids=ids)
    st.success(f" ✅ Added {len(docs)} songs to database.")

# -----------------------------
# RETRIEVAL + GENERATION (STREAMING)
# -----------------------------
def retrieve_lyrics(query, top_k=3):
    """Retrieve similar lyrics from ChromaDB."""
    collection = get_chroma_collection()
    if collection.count() == 0:
        st.error("Database is empty. Please upload a JSON and build the DB.")
        return []
        
    results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas"])
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    return list(zip(docs, metas))

def generate_with_ollama_stream(prompt, context, model=MODEL_DEFAULT, max_tokens=150):
    """
    Generates new lyrics using the Ollama Python API and STREAMS the response.
    This function returns a GENERATOR.
    """
    full_prompt = (
        "You are a creative lyricist AI.\n\n"
        "Use the following examples of lyrics for stylistic reference, but do NOT copy them directly.\n\n"
        f"--- EXAMPLES ---\n{context}\n\n"
        f"--- TASK ---\nWrite an original song based on this theme: {prompt}\n"
    )

    try:
        # 'stream=True' makes this return a generator
        stream = ollama.generate(
            model=model,
            prompt=full_prompt,
            stream=True,
            options={
                "num_predict": max_tokens, # Renamed from max_tokens in the API
                "temperature": 0.8
            }
        )

        # Yield each part of the response as it comes in
        for chunk in stream:
            if 'response' in chunk:
                yield chunk['response']

    except ollama.ResponseError as e:
        yield f" ⚠️ Ollama API Error: {e.error}"
    except Exception as e:
        yield f" ⚠️ Unexpected error: {str(e)}"

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title=" 🎵 AI Song Lyric Generator", page_icon=" 🎤 ", layout="centered")
st.title(" 🎵 AI Song Lyric Generator (RAG App)")
st.caption("Offline RAG demo using Chroma + Ollama LLM")

st.divider()

# --- 1. Database Section ---
with st.expander(" 🗂️ Step 1: Upload & Build Database", expanded=True):
    uploaded_file = st.file_uploader("Upload a JSON file containing song objects (title, artist, lyrics)", type=["json"])

    if uploaded_file is not None:
        # Save the file temporarily to be read by the builder
        os.makedirs(DATA_PATH, exist_ok=True)
        save_path = os.path.join(DATA_PATH, uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button(" 📦 Build Lyric Database"):
            with st.spinner("Embedding lyrics... this may take a moment."):
                build_db_from_json(save_path)
            # Clean up the temp file
            os.remove(save_path)

st.divider()

# --- 2. Generation Section ---
st.header(" 🎤 Step 2: Generate New Lyrics")
prompt = st.text_input("Enter a theme or prompt (e.g. 'Love and the Ocean')")

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.number_input("Retrieved Lyrics (k)", 1, 10, 3)
with col2:
    model_name = st.text_input("Ollama Model", MODEL_DEFAULT)
with col3:
    max_tokens = st.slider("Max Tokens", 50, 500, 150)

if st.button(" 🎶 Generate Lyrics"):
    if not prompt.strip():
        st.warning("Please enter a prompt or theme.")
    else:
        try:
            results = retrieve_lyrics(prompt, top_k=top_k)
            
            if results:
                context_text = "\n\n".join([f"{m['artist']} - {m['title']}\n{d}" for d, m in results])
                
                # --- This is the new streaming logic ---
                st.subheader(" ✨ Generated Lyrics")
                with st.container(border=True):
                    # 1. Get the streaming generator
                    streaming_response = generate_with_ollama_stream(
                        prompt, 
                        context_text, 
                        model=model_name, 
                        max_tokens=max_tokens
                    )
                    
                    # 2. Use st.write_stream to render the output in real-time
                    # This will also capture the full text when it's done
                    full_lyrics = st.write_stream(streaming_response)
                
                # --- Show retrieved lyrics AFTER generation ---
                st.subheader(" 📚 Reference Lyrics Used")
                for doc, meta in results:
                    with st.expander(f"{meta['artist']} - {meta['title']}"):
                        st.write(doc)

                # --- Add download button for the captured full text ---
                st.download_button(
                    " 💾 Download Lyrics", 
                    full_lyrics, 
                    file_name="generated_lyrics.txt"
                )

        except Exception as e:
            st.error(f"Error during generation: {e}")

st.divider()
st.caption(" 🧠 Powered by Ollama (local LLM) + ChromaDB (vector store) + Streamlit UI")