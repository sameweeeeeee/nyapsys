import os
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_HOST = os.getenv("CHROMA_HOST", "http://127.0.0.1:8001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))

client = None
collection = None
embedding_model = None


def init_collection():
    global client, collection, embedding_model
    client = chromadb.Client(settings=chromadb.config.Settings(chroma_server_host="127.0.0.1", chroma_server_http_port=8001))
    try:
        collection = client.get_collection("nyapsys_kb")
    except:
        collection = client.create_collection("nyapsys_kb", metadata={"hnsw:space": "cosine"})
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_and_upsert(chunks, metadatas, ids):
    if not collection or not embedding_model:
        raise RuntimeError("Collection not initialized")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    collection.upsert(embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas, ids=ids)


def query(text, conversation_id=None, top_k=RAG_TOP_K):
    if not collection or not embedding_model:
        raise RuntimeError("Collection not initialized")
    query_embedding = embedding_model.encode([text], show_progress_bar=False)
    where = {"conversation_id": conversation_id} if conversation_id else None
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k, where=where)
    matches = []
    for i, doc in enumerate(results["documents"][0]):
        score = 1 - results["distances"][0][i]
        if score >= RAG_SCORE_THRESHOLD:
            matches.append({"id": results["ids"][0][i], "document": doc, "score": score, "metadata": results["metadatas"][0][i]})
    return matches


def delete_by_conversation(conversation_id: str):
    if not collection:
        raise RuntimeError("Collection not initialized")
    results = collection.get(where={"conversation_id": conversation_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])