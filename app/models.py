"""Data shapes that move between modules.

Pydantic models = typed, validated objects. If a field is missing or has the
wrong type, we get a clear error instead of a silent bug later.
"""
from pydantic import BaseModel, Field


class Page(BaseModel):
    doc_id: str
    doc_name: str
    page: int  # 1-based, like a human reads it
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    section: str  # e.g. "8. Obligations of Data Fiduciary"
    chunk_index: int
    text: str
    owner: str = ""  # "" = public library; otherwise the session id of the user who uploaded it
    created_at: float = 0.0  # unix time, used to auto-delete user uploads


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    source: str  # "bm25" | "vector" | "hybrid" | "rerank"


class Citation(BaseModel):
    doc_name: str
    page: int
    section: str
    snippet: str
    url: str = ""  # link to the public PDF page ("" for private user uploads)


class Answer(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[Citation] = Field(default_factory=list)
    search_query: str = ""  # what we actually searched for (after rewriting)


class LLMAnswer(BaseModel):
    """The exact JSON shape we force the LLM to return (structured output)."""

    found: bool = Field(description="true only if the sources contain the answer")
    answer: str = Field(description="the answer, in simple words")
    source_ids: list[str] = Field(description='ids of sources used, like ["S1", "S3"]')


# ---------------- Problem solver ("I have a problem, what do I do?") ----------------
class LLMQueries(BaseModel):
    """Step 1: the LLM splits a messy problem into a few focused search queries."""

    queries: list[str] = Field(description="1 to 3 short formal English search queries")


class LLMPlanPoint(BaseModel):
    text: str = Field(description="one short point in simple words")
    source_ids: list[str] = Field(description='sources that support this point, like ["S2"]')


class LLMPlan(BaseModel):
    """Step 2: the exact JSON shape we force the LLM to return for an action plan."""

    found: bool = Field(description="true only if the sources help with this problem")
    summary: str = Field(description="1-2 lines: what the situation is under these rules")
    rights: list[LLMPlanPoint] = Field(description="the user's rights / what the law says")
    steps: list[LLMPlanPoint] = Field(description="what the user should do now, in order")
    escalation: list[LLMPlanPoint] = Field(description="where to complain if it is not solved")
    draft_letter: str = Field(description="short complaint/request letter, or empty")
    missing_info: list[str] = Field(description="facts the user should tell us for a better plan")


class PlanPoint(BaseModel):
    text: str
    refs: list[int] = Field(default_factory=list)  # 1-based numbers into ActionPlan.citations


class ActionPlan(BaseModel):
    problem: str
    found: bool
    summary: str = ""
    rights: list[PlanPoint] = Field(default_factory=list)
    steps: list[PlanPoint] = Field(default_factory=list)
    escalation: list[PlanPoint] = Field(default_factory=list)
    draft_letter: str = ""
    missing_info: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
