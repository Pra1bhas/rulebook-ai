from app import llm
from app.config import settings
from app.generation.answer import drop_invented_numbers


# --- Bug 1: rewrite invented "Section 6" and the reranker dropped the right chunk ---
def test_rewrite_drops_section_number_user_never_typed():
    q = "Can I withdraw my consent?"
    assert drop_invented_numbers(q, "DPDP Act withdrawal of consent Section 6") == (
        "DPDP Act withdrawal of consent"
    )


def test_rewrite_keeps_section_number_user_typed():
    q = "What does section 8(6) say about breach?"
    assert drop_invented_numbers(q, "Section 8(6) personal data breach intimation") == (
        "Section 8(6) personal data breach intimation"
    )


# --- Bug 2: Gemini 503 "high demand" crashed instead of retrying / falling back ---
def test_gemini_busy_retries_then_falls_back_to_groq(monkeypatch):
    calls = {"gemini": 0}

    def busy_gemini(prompt, schema):
        calls["gemini"] += 1
        raise llm.RateLimited()

    monkeypatch.setattr(settings, "llm_provider", "gemini")
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_max_retries", 2)
    monkeypatch.setattr(llm, "_gemini", busy_gemini)
    monkeypatch.setattr(llm, "_groq", lambda prompt, schema: "groq answer")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # don't really wait in tests

    assert llm.generate("hi") == "groq answer"
    assert calls["gemini"] == 3  # 1 try + 2 retries before falling back


def test_503_is_treated_as_retryable():
    assert 503 in llm.RETRYABLE_CODES and 429 in llm.RETRYABLE_CODES
