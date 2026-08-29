"""Tests for the cite-or-retry generation guard."""

from app.rag import generate_answer
from app.schemas import Citation


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        self.response_metadata = {"model_name": "test-model"}


class ScriptedLLM:
    """Returns queued responses in order; records how many times it was called."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _Resp(self._contents.pop(0))


CITED = "The CSF has a Core (nist_csf.pdf, p.7)."
UNCITED = "The CSF has a Core, Profiles, and Tiers."


def _c():
    return [Citation(document="nist_csf.pdf", page=7, snippet="t")]


def test_no_retry_when_first_answer_is_cited():
    llm = ScriptedLLM([CITED])
    answer, usage, *_ = generate_answer("q", _c(), retry_uncited=True, llm=llm)
    assert answer == CITED
    assert llm.calls == 1  # no retry needed


def test_retry_recovers_citations_when_first_answer_is_uncited():
    llm = ScriptedLLM([UNCITED, CITED])
    answer, usage, *_ = generate_answer("q", _c(), retry_uncited=True, llm=llm)
    assert answer == CITED  # the cited retry is used
    assert llm.calls == 2
    # both calls' tokens are counted
    assert usage.input_tokens == 20 and usage.output_tokens == 10


def test_retry_disabled_by_default_keeps_uncited_answer():
    llm = ScriptedLLM([UNCITED])
    answer, _u, *_ = generate_answer("q", _c(), llm=llm)  # retry_uncited defaults False
    assert answer == UNCITED
    assert llm.calls == 1


def test_retry_gives_up_after_one_extra_attempt():
    """If the retry also fails to cite, keep the first answer (don't loop)."""
    llm = ScriptedLLM([UNCITED, UNCITED])
    answer, _u, *_ = generate_answer("q", _c(), retry_uncited=True, llm=llm)
    assert answer == UNCITED
    assert llm.calls == 2  # exactly one retry, then stop
