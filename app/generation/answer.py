"""Turn the best chunks into a grounded answer with citations.

Three guards against hallucination (AI making things up):
  1. The prompt says: answer ONLY from these sources, else found=false.
  2. Structured output: the LLM must return JSON {found, answer, source_ids}.
  3. We verify every source_id really exists. Fake ids are dropped, and if
     nothing valid is left, we say "not found" instead of trusting the text.
"""
import re
from urllib.parse import quote

from app import llm
from app.models import Answer, Citation, LLMAnswer, SearchResult

NOT_FOUND = (
    "I could not find this in the uploaded documents. "
    "Please check the official source or ask a more specific question."
)

ANSWER_PROMPT = """You are RuleBook AI. You explain Indian rules and regulations to normal people.

Rules:
- Use ONLY the sources below. Do not use outside knowledge.
- If the sources do not contain the answer, set found=false and leave answer empty.
- Write in simple words, short (max ~120 words). Use bullet points if it helps.
- Reply in the same language style as the question (English, or Telugu written in English letters).
- In source_ids list every source you used, e.g. ["S1", "S3"].

Sources:
{sources}

Question: {question}
"""

REWRITE_PROMPT = """Rewrite the user's question as a short, formal English search query
for searching Indian legal/regulatory documents (RBI, SEBI, DPDP Act).
The question may be in Telugu written in English letters. Keep section numbers as they are.
Do NOT add section numbers, act names, years or topics the user did not mention.
Return ONLY the query, nothing else.

Question: {question}"""

# "Section 6", "sub-section (6)", "Rule 3(1)", or a bare number like "2023"
_NUMBER_REF = re.compile(r"\b(?:sub-?section|section|sec\.?|rule|clause|regulation)?\s*\(?\d+[\w()]*", re.I)


def drop_invented_numbers(question: str, query: str) -> str:
    """Remove numbers the LLM added that the user never typed.

    Bug we hit: "Can I withdraw my consent?" was rewritten to "...Section 6", and the
    reranker then preferred chunks mentioning "sub-section (6)" over the real answer.
    """
    asked = set(re.findall(r"\d+", question))

    def keep(m: re.Match) -> str:
        return m.group(0) if set(re.findall(r"\d+", m.group(0))) <= asked else ""

    return " ".join(_NUMBER_REF.sub(keep, query).split())


def rewrite_query(question: str) -> str:
    """'Bank naa data share cheyyocha?' -> 'Can a bank disclose customer personal data?'"""
    try:
        out = llm.generate(REWRITE_PROMPT.format(question=question))
        out = drop_invented_numbers(question, str(out).strip().strip('"'))
        return out or question
    except llm.LLMError:
        return question  # search with the original question if rewriting fails


def build_sources(results: list[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        parts.append(f"[S{i}] ({c.doc_name}, page {c.page}, {c.section})\n{c.text}")
    return "\n\n".join(parts)


def answer_question(question: str, results: list[SearchResult], search_query: str = "") -> Answer:
    if not results:
        return Answer(question=question, answer=NOT_FOUND, found=False, search_query=search_query)

    prompt = ANSWER_PROMPT.format(sources=build_sources(results), question=question)
    out: LLMAnswer = llm.generate(prompt, schema=LLMAnswer)

    # Guard 3: keep only citations that point to real sources we sent.
    valid = {f"S{i}": r for i, r in enumerate(results, start=1)}
    used = [valid[s] for s in dict.fromkeys(out.source_ids) if s in valid]

    if not out.found or not used or not out.answer.strip():
        return Answer(question=question, answer=NOT_FOUND, found=False, search_query=search_query)

    return Answer(
        question=question,
        answer=out.answer.strip(),
        found=True,
        citations=make_citations(used)[0],
        search_query=search_query,
    )


def make_citations(used: list[SearchResult]) -> tuple[list[Citation], dict[str, int]]:
    """Turn chunks into citations, merging chunks from the same doc + page + section.

    Two chunks of one section used to show up as two identical "p5, Section 6" lines.
    Returns (citations, chunk_id -> 1-based citation number).
    """
    citations: list[Citation] = []
    number: dict[tuple, int] = {}
    by_chunk: dict[str, int] = {}
    for r in used:
        c = r.chunk
        key = (c.doc_id, c.page, c.section)
        if key not in number:
            citations.append(
                Citation(
                    doc_name=c.doc_name,
                    page=c.page,
                    section=c.section,
                    snippet=c.text[:300],
                    # Public PDFs are served at /files, so the link opens the exact page.
                    url="" if c.owner else f"files/{quote(c.doc_name)}#page={c.page}",
                )
            )
            number[key] = len(citations)
        by_chunk[c.chunk_id] = number[key]
    return citations, by_chunk
