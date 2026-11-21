# NHTSA Quality Agent

> 🚧 Work in progress

A RAG assistant and agent for automotive quality incidents, built on public
[NHTSA](https://www.nhtsa.gov/) vehicle complaints and recalls:

- **RAG**: ask a question in natural language, get an answer with sources.
- **Agent**: classify a complaint by component, find related recalls, summarize recurring defects for a model.
- **Evaluation**: test set and metrics for classification accuracy and retrieval relevance.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv and install locked dependencies
cp .env.example .env    # then fill in your keys
uv run pytest           # run the test suite
```

## Usage

```bash
uv run python -m nhtsa_quality_agent.data.ingest   # download raw complaints & recalls (~1 min)
uv run python -m nhtsa_quality_agent.data.clean    # clean into data/processed/*.parquet
uv run python -m nhtsa_quality_agent.rag.index     # embed & index into data/index (Chroma)
uv run python -m nhtsa_quality_agent.rag.ask "What do owners report about phantom braking?" --model "model 3"
```

`ask` accepts `--make`, `--model`, `--year`, `--source {complaint,recall}`, `-k` and
`--retrieve-only` (no LLM call). Answers cite their sources as `[C-<odi number>]`
(complaint) or `[R-<campaign number>]` (recall).

## How it works

```
NHTSA API ──ingest──▶ data/raw (JSON) ──clean──▶ data/processed (Parquet)
    ──index──▶ Chroma (bge-small-en-v1.5 embeddings, cosine)
    ──ask──▶ top-k sources (+ metadata filters) ──▶ Claude ──▶ answer with citations
```

- **One complaint = one document** (median ~90 words, no chunking needed). Recalls are
  indexed once per covered vehicle so they can be filtered by make/model/year, then
  deduplicated by campaign at query time.
- **Embeddings run locally** with [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch).
- **Grounding**: the model must cite source ids; citations that don't match a retrieved
  source are flagged as hallucinated.

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
```

## Project structure

```
src/nhtsa_quality_agent/   # application package
tests/                     # pytest suite
data/                      # local data (git-ignored)
notebooks/                 # exploration
```
