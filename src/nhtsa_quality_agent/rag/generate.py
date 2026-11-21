"""Generate a sourced answer from retrieved documents with Claude."""

import re
from dataclasses import dataclass
from typing import Any

from nhtsa_quality_agent.rag.retrieve import Hit

SYSTEM_PROMPT = """\
You are a vehicle-safety analyst. You answer questions about US vehicle complaints and \
recalls using ONLY the sources provided in the user message (NHTSA complaints and recalls).

Rules:
- Every factual sentence must cite at least one source with its id in square brackets, \
e.g. [C-11777952] or [R-20V682000]. Cite only ids that appear in the sources.
- If the sources do not contain the answer, say so plainly instead of guessing.
- Complaints are unverified owner reports: describe them as reports, not as proven defects.
- Be concise: a short direct answer, then the supporting points.\
"""

CITATION_PATTERN = re.compile(r"\[([CR]-[A-Z0-9]+)\]")


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Hit]  # sources actually cited, in order of first citation
    unknown_citations: list[str]  # cited ids that were NOT in the context (hallucinated)


def format_context(hits: list[Hit]) -> str:
    blocks = [f'<source id="{hit.citation}">\n{hit.text}\n</source>' for hit in hits]
    return "\n\n".join(blocks)


def build_user_message(question: str, hits: list[Hit]) -> str:
    # Sources first, question last: long context before the instruction works best.
    return f"<sources>\n{format_context(hits)}\n</sources>\n\nQuestion: {question}"


def extract_citations(text: str) -> list[str]:
    """Cited ids in order of first appearance, without duplicates."""
    return list(dict.fromkeys(CITATION_PATTERN.findall(text)))


def generate_answer(
    client: Any,  # anthropic.Anthropic, typed loosely so tests can pass a fake
    model: str,
    question: str,
    hits: list[Hit],
    max_tokens: int = 4096,
) -> Answer:
    if not hits:
        return Answer("No matching complaints or recalls were found.", [], [])

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},  # short grounded answer: no need for deep reasoning
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    )
    if response.stop_reason == "refusal":
        return Answer("The model declined to answer this question.", [], [])

    text = "".join(block.text for block in response.content if block.type == "text").strip()
    by_citation = {hit.citation: hit for hit in hits}
    cited = extract_citations(text)
    return Answer(
        text=text,
        sources=[by_citation[c] for c in cited if c in by_citation],
        unknown_citations=[c for c in cited if c not in by_citation],
    )
