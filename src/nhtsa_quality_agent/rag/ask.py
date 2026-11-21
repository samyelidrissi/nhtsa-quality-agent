"""Ask a question about NHTSA complaints and recalls, get a sourced answer.

Examples:
  uv run python -m nhtsa_quality_agent.rag.ask "Why do owners complain about phantom braking?"
  uv run python -m nhtsa_quality_agent.rag.ask "What fuel pump issues?" --make toyota --year 2020
  uv run python -m nhtsa_quality_agent.rag.ask "Brake problems" --model "model 3" --retrieve-only
"""

import argparse

import anthropic
import chromadb
from dotenv import load_dotenv

from nhtsa_quality_agent.config import DEFAULT_DATA_DIR, load_settings
from nhtsa_quality_agent.rag.embeddings import FastEmbedEmbedder
from nhtsa_quality_agent.rag.generate import generate_answer
from nhtsa_quality_agent.rag.index import get_collection
from nhtsa_quality_agent.rag.retrieve import Filters, Hit, Retriever


def _print_hits(hits: list[Hit]) -> None:
    for hit in hits:
        meta = hit.metadata
        vehicle = f"{meta['make']} {meta['model']} {meta['model_year']}"
        preview = hit.text.splitlines()[-1][:110]
        print(f"  [{hit.citation}] {hit.score:.2f}  {vehicle}  {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--make")
    parser.add_argument("--model")
    parser.add_argument("--year", type=int)
    parser.add_argument("--source", choices=["complaint", "recall"])
    parser.add_argument("-k", type=int, default=8, help="number of sources to retrieve")
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM call")
    args = parser.parse_args()
    load_dotenv()

    client = chromadb.PersistentClient(path=str(DEFAULT_DATA_DIR / "index"))
    retriever = Retriever(get_collection(client), FastEmbedEmbedder())
    filters = Filters(make=args.make, model=args.model, model_year=args.year, source=args.source)
    hits = retriever.search(args.question, k=args.k, filters=filters)

    print(f"\nRetrieved {len(hits)} sources:")
    _print_hits(hits)
    if args.retrieve_only:
        return

    settings = load_settings()
    llm = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    answer = generate_answer(llm, settings.llm_model, args.question, hits)

    print(f"\nAnswer ({settings.llm_model}):\n{answer.text}\n")
    print(f"Cited {len(answer.sources)}/{len(hits)} retrieved sources.")
    if answer.unknown_citations:
        print(f"WARNING: cited ids not in the sources: {answer.unknown_citations}")


if __name__ == "__main__":
    main()
