import argparse
import logging
import json
import os
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in your environment")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_RETRIEVAL_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 500
RETRIEVAL_K = 12
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "movie_plots"

client = OpenAI(api_key=OPENAI_API_KEY)


def load_and_preprocess_data(file_path: str, num_rows: int = 10) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df[["Title", "Plot"]].dropna().head(num_rows)
    logger.info(f"Loaded {len(df)} rows from dataset")
    return df


def chunk_text(title: str, text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_body = " ".join(words[i:i + chunk_size])
        chunk = f"Title: {title}\nPlot: {chunk_body}"
        logger.info(f"Chunking '{chunk}' ..")
        chunks.append(chunk)
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    logger.info(f"Embedding {len(texts)} chunks")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in response.data]


def build_chroma_collection(df: pd.DataFrame) -> chromadb.Collection:
    os.makedirs(CHROMA_PATH, exist_ok=True)

    embedding_function = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    if collection.count() > 0:
        logger.info(f"Using existing Chroma collection with {collection.count()} documents")
        return collection

    logger.info("Building Chroma collection from scratch...")
    for idx, row in df.iterrows():
        chunks = chunk_text(row["Title"], row["Plot"])
        embeddings = embed_texts(chunks)
        collection.add(
            ids=[f"{idx}_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"title": row["Title"], "chunk_id": i} for i in range(len(chunks))]
        )

    logger.info(f"Chroma indexing complete ({collection.count()} chunks)")
    return collection


def retrieve_chroma(query: str, collection: chromadb.Collection) -> List[str]:
    results = collection.query(
        query_texts=[query],
        n_results=RETRIEVAL_K
    )
    retrieved = results["documents"][0]
    # logger.info(f"Retrieved for {query}: {retrieved}")
    logger.info(f"Retrieved {len(retrieved)} chunks for query: {query}")
    return retrieved


def generate_answer(query: str, contexts: List[str]) -> Dict:
    if not contexts:
        return {"answer": "No relevant information found.", "contexts": [], "explanation": "Vector search returned no results."}

    prompt = f"""
You are a movie expert assistant. Woohoo!

Your task:
- Answer the question using ONLY the provided contexts.
- Do NOT include any information not present in the contexts.
- Provide a structured JSON output with the following keys:
  1. "answer" (short, natural language)
  2. "contexts" (list of the relevant snippets used)
  3. "explanation" (brief reasoning of how you arrived at the answer)

Contexts:
{chr(10).join(contexts)}

Question:
{query}

Return ONLY valid JSON conforming to the above schema.
"""
    response = client.chat.completions.create(
        model=LLM_RETRIEVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "explanation": {"type": "string"}
                    },
                    "required": ["answer", "explanation"]
                }
            }
        }
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from model")
        return {"answer": "Model output parsing failed.", "contexts": contexts, "explanation": raw}

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline for querying movie plots")
    parser.add_argument("--query", required=True)
    parser.add_argument("--rows", type=int, default=500)
    args = parser.parse_args()

    df = load_and_preprocess_data("./data/wiki_movie_plots_deduped.csv", args.rows)
    collection = build_chroma_collection(df)
    contexts = retrieve_chroma(args.query, collection)
    output = generate_answer(args.query, contexts)    
    output["contexts"] = contexts
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()