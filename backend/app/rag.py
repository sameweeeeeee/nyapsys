import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


CHROMA_HOST = os.getenv("CHROMA_HOST", "http://chromadb:8000")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "nyapsys_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))


client: Optional[chromadb.Client] = None
collection: Optional[chromadb.Collection] = None
embedding_model: Optional[SentenceTransformer] = None


def init_collection():
    global client, collection, embedding_model

    client = chromadb.Client(
        Settings(
            chroma_server_host=CHROMA_HOST.split(":")[0],
            chroma_server_http_port=int(CHROMA_HOST.split(":")[1])
            if ":" in CHROMA_HOST
            else 8000,
        )
    )

    try:
        collection = client.get_collection(name=CHROMA_COLLECTION)
    except Exception:
        collection = client.create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_and_upsert(
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    if not collection or not embedding_model:
        raise RuntimeError("Collection not initialized. Call init_collection() first.")

    embeddings = embedding_model.encode(chunks, show_progress_bar=False)

    collection.upsert(
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )


def query(
    text: str,
    conversation_id: Optional[str] = None,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    if not collection or not embedding_model:
        raise RuntimeError("Collection not initialized. Call init_collection() first.")

    query_embedding = embedding_model.encode([text], show_progress_bar=False)

    where = None
    if conversation_id:
        where = {"conversation_id": conversation_id}

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=where,
    )

    matches = []
    for i, doc in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        score = 1 - distance

        if score >= RAG_SCORE_THRESHOLD:
            matches.append(
                {
                    "id": results["ids"][0][i],
                    "document": doc,
                    "score": score,
                    "metadata": results["metadatas"][0][i],
                }
            )

    return matches


def delete_by_file_id(file_id: str) -> None:
    if not collection:
        raise RuntimeError("Collection not initialized. Call init_collection() first.")

    results = collection.get(where={"file_id": file_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def delete_by_conversation(conversation_id: str) -> None:
    if not collection:
        raise RuntimeError("Collection not initialized. Call init_collection() first.")

    results = collection.get(where={"conversation_id": conversation_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])