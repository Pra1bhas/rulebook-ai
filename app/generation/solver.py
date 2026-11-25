"""Problem solver: "I have this problem, what do I do?" -> a cited action plan.

Two LLM calls:
  1. plan_queries: split a messy, real-life problem into 1-3 focused search queries.
     "UPI fraud lo 20k poyayi, bank reply ivvatledu" ->
       ["customer liability unauthorised electronic banking transaction",
        "complaint against bank not resolved ombudsman"]
  2. solve_problem: write rights / steps / where to complain / draft letter,
     using ONLY the retrieved sources.

Same hallucination guards as answer.py, applied to EVERY point:
  a point with no valid source id is dropped. If nothing grounded is left -> found=False.
  (The draft letter and "missing info" are writing help, so they have no citations.)
"""
from app import llm
from app.generation.answer import build_sources, drop_invented_numbers, make_citations
from app.models import ActionPlan, LLMPlan, LLMPlanPoint, LLMQueries, PlanPoint, SearchResult

QUERIES_PROMPT = """A person in India describes a problem. It may be in Telugu written in English letters.
Write 1 to 3 short, formal English search queries to find the rules that apply
(RBI, SEBI, consumer protection, data protection, RTI, insurance...).
One query per separate issue. Do NOT add section numbers or details the person did not say.

Problem: {problem}"""

SOLVE_PROMPT = """You are RuleBook AI. You help normal people in India solve problems using official rules.

Rules:
- Use ONLY the sources below. No outside knowledge. Do not invent phone numbers, websites,
  amounts or deadlines that are not in the sources.
- If the sources do not help with this problem, set found=false and leave the lists empty.
- Every point in rights, steps and escalation must list the source ids that support it.
- steps: concrete actions in order, starting with the most urgent (mention deadlines from the sources).
- escalation: where to complain if the first steps do not work (only if the sources say so).
- draft_letter: a short, polite letter the person can send (fill unknowns as [Your Name], [Date]...).
  Leave it empty if no letter is needed.
- missing_info: up to 3 facts that would change the advice (e.g. "When did the fraud happen?").
- Simple words, short points. Reply in the same language style as the problem
  (English, or Telugu written in English letters). Keep the letter in English.

Sources:
{sources}

Problem: {problem}"""


def plan_queries(problem: str) -> list[str]:
    try:
        out: LLMQueries = llm.generate(QUERIES_PROMPT.format(problem=problem), schema=LLMQueries)
        queries = [drop_invented_numbers(problem, q.strip()) for q in out.queries]
        queries = [q for q in dict.fromkeys(queries) if q][:3]
        return queries or [problem]
    except llm.LLMError:
        return [problem]  # still search with the user's own words


def solve_problem(problem: str, results: list[SearchResult], queries: list[str]) -> ActionPlan:
    if not results:
        return ActionPlan(problem=problem, found=False, search_queries=queries)

    prompt = SOLVE_PROMPT.format(sources=build_sources(results), problem=problem)
    out: LLMPlan = llm.generate(prompt, schema=LLMPlan)

    valid = {f"S{i}": r for i, r in enumerate(results, start=1)}
    sections = {"rights": out.rights, "steps": out.steps, "escalation": out.escalation}

    # Collect every source actually used, in order, so citation numbers are stable.
    used_ids = [s for pts in sections.values() for p in pts for s in p.source_ids if s in valid]
    used = [valid[s] for s in dict.fromkeys(used_ids)]
    if not out.found or not used:
        return ActionPlan(problem=problem, found=False, search_queries=queries)
    citations, by_chunk = make_citations(used)

    def grounded(points: list[LLMPlanPoint]) -> list[PlanPoint]:
        kept = []
        for p in points:
            refs = sorted({by_chunk[valid[s].chunk.chunk_id] for s in p.source_ids if s in valid})
            if refs and p.text.strip():  # no valid source -> we don't trust this point
                kept.append(PlanPoint(text=p.text.strip(), refs=refs))
        return kept

    return ActionPlan(
        problem=problem,
        found=True,
        summary=out.summary.strip(),
        rights=grounded(out.rights),
        steps=grounded(out.steps),
        escalation=grounded(out.escalation),
        draft_letter=out.draft_letter.strip(),
        missing_info=[m.strip() for m in out.missing_info if m.strip()][:3],
        citations=citations,
        search_queries=queries,
    )
