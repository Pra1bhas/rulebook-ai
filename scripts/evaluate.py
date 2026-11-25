"""Give the system an exam and get real numbers for your resume.

data/eval/questions.json has questions + where the answer really is
(the section number it must come from). We measure:

  Retrieval hit rate @5 : % of questions where the correct section is in the top 5 chunks
  MRR                   : how HIGH the correct chunk is ranked (1.0 = always first)
  Not-found accuracy    : for questions NOT in the docs, % where we correctly say "not found"
                          (only with --llm, uses LLM calls)

Usage:
  python -m scripts.evaluate                 # retrieval only, no LLM calls, free
  python -m scripts.evaluate --llm           # also tests "not found" behaviour

Try: change chunk size / turn reranker off (RERANKER=none) and compare numbers.
"""
import json
import re
import sys
from pathlib import Path

from app.pipeline import RuleBook

EVAL_FILE = Path("data/eval/questions.json")


def section_matches(section: str, text: str, expected: str) -> bool:
    """expected like '8' matches a chunk whose section heading starts with '8.'"""
    pattern = rf"^\s*{re.escape(expected)}\.\s"
    return bool(re.match(pattern, section)) or bool(re.match(pattern, text))


def main(use_llm: bool) -> None:
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    rb = RuleBook()
    hits, rr_total, rows = 0, 0.0, []

    for item in data["answerable"]:
        results = rb.search(item["question"])
        rank = next(
            (i for i, r in enumerate(results, 1)
             if section_matches(r.chunk.section, r.chunk.text, item["expected_section"])),
            None,
        )
        hits += rank is not None
        rr_total += 1 / rank if rank else 0
        rows.append((item["question"][:55], item["expected_section"], rank or "-"))

    n = len(data["answerable"])
    print(f"\n{'Question':57} {'Sec':>4} {'Rank':>5}")
    for q, s, r in rows:
        print(f"{q:57} {s:>4} {r!s:>5}")
    print(f"\nRetrieval hit rate @5: {hits / n:.0%}  ({hits}/{n})")
    print(f"MRR: {rr_total / n:.2f}")

    if use_llm:
        correct = 0
        for q in data["not_in_docs"]:
            ans = rb.ask(q)
            correct += not ans.found
            print(f"  not-found test: {'PASS' if not ans.found else 'FAIL'}  {q}")
        print(f"Not-found accuracy: {correct}/{len(data['not_in_docs'])}")


if __name__ == "__main__":
    main(use_llm="--llm" in sys.argv)
