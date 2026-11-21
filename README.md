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
