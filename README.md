# Mini RAG System (Movie Plots)


Super lightweight RAG pipeline for querying movie plots based on the [Kaggle Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset.

## Requirements

- Python 3.11–3.12
- OpenAI API key  
- [`uv`](https://uv.dev/) for environment and dependency management  

## Setup

1. **Install Python version and create environment**

```bash
uv python install 3.11
uv venv --python 3.11
uv sync
```

2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-proj..."
```

3. The wiki_movie_plots_deduped.csv should be placed in the data directory (if not alraedy). The CSV must contain Title and Plot columns. You can download the CSV from [https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).


## Usage

To run the system:

```bash
uv run python main.py --query "Tell me about The Matrix"
```

### Arguments

- `--query` (required) – your question about movie plots
- `--rows` (optional) – number of movies to index (default 500)

## Pipeline Overview

- Ingestion: Loads the CSV dataset, subsets up to 500 rows (configurable), drops missing data, and chunks each plot (~500 words/chunk) while including the movie title in each chunk.
- Embedding/Storage: Uses OpenAI embeddings (`text-embedding-3-small`) to convert each chunk into vectors.  
  - ChromaDB is the default persistent storage, storing embeddings + metadata (`title`, `chunk_id`) on disk.  
- Retrieval: 
  1. Query is embedded using the same embedding model.  
  2. Top `k=12` most semantically similar chunks are retrieved from the vector store.
- Generation: LLM (`gpt-4.1-mini`) receives the retrieved contexts and the query.  
  - Prompt enforces **use only provided contexts**.  
  - Model outputs a structured JSON with:
    - `answer` (short natural language answer)  
    - `contexts` (relevant chunks used)  
    - `explanation` (brief reasoning)
- **Output**: JSON object printed to console or returned to calling program.  