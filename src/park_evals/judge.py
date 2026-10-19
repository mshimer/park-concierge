"""LLM-as-judge: score one answer against one scenario's rubric."""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from park_core.config import settings

log = logging.getLogger(__name__)

JUDGE_SYSTEM = """You grade a park concierge assistant's answer against a rubric. You are strict
and literal.

Return one ruling per rubric criterion, in the order given. Judge each criterion INDEPENDENTLY,
against its own words only - do not let a criterion you judged harshly drag down the next one, and
do not import requirements from one criterion into another.

Rules:
- Judge ONLY what the criterion says. Ignore style, length and tone.
- Read quantifiers exactly. "at least two" is met by two; do not demand all. "either X or Y" is met
  by X alone; do not demand both.
- A criterion that describes a property of the recommendations is met when the recommendations have
  that property. Do not additionally require the answer to comment on it.
- The retrieved context is the ONLY source of truth about parks. A park fact absent from the context
  is ungrounded, and a park named in the answer but missing from the context is a hallucination.
- `met` is true only if the criterion is clearly satisfied. In `reason`, quote or paraphrase the
  part of the answer that decided it. One sentence.
"""


class CriterionVerdict(BaseModel):
    """How the answer fared against one rubric criterion."""

    criterion: str = Field(description="the criterion being judged, copied from the rubric")
    met: bool = Field(description="true only if this criterion is clearly satisfied")
    reason: str = Field(description="the part of the answer that decided it, one sentence")


class Verdict(BaseModel):
    """A judge's structured ruling, one entry per rubric criterion.

    CONFUSING-BIT WARNING: this started out as a single holistic {passed, score, reason}, which read
    cleanly but graded badly. Asked to weigh three or four criteria at once, the judge reliably lost
    track of quantifiers - it would read "for at least two of its suggestions" as "for all of them",
    and "either the park is under the limit or the answer flags it" as "both" - and failed answers
    that were in fact correct. Grading one criterion at a time, with the criterion text restated in
    the output, made the verdicts stable. The score is then arithmetic rather than a vibe.
    """

    criteria: list[CriterionVerdict] = Field(description="one ruling per rubric criterion, in order")

    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.met for c in self.criteria)

    @property
    def score(self) -> int:
        """0-5, proportional to the criteria met."""
        if not self.criteria:
            return 0
        return round(5 * sum(c.met for c in self.criteria) / len(self.criteria))

    @property
    def reason(self) -> str:
        """The criteria that were not met, or a note that all were."""
        unmet = [c for c in self.criteria if not c.met]
        if not unmet:
            return f"all {len(self.criteria)} criteria met"
        return "; ".join(f"UNMET [{c.criterion}]: {c.reason}" for c in unmet)


def default_judge(**kwargs: Any) -> BaseChatModel:
    """A different, cheaper model than the agent.

    Deliberately not the agent's own model: a judge sharing the agent's weights tends to find its
    reasoning persuasive and rubber-stamp it.
    """
    from park_core.models import make_chat_model

    return make_chat_model(settings.judge_model, **kwargs)


@traceable(name="judge_answer", run_type="chain", tags=["eval", "judge"])
def judge_answer(
    prompt: str,
    answer: str,
    retrieved: list[dict[str, Any]],
    rubric: list[str],
    *,
    model: BaseChatModel | None = None,
) -> Verdict:
    """Grade one answer. Returns the judge's structured verdict."""
    model = model or default_judge()
    # Trim the context: the judge only needs what identifies and bounds each park, not every
    # matched chunk in full, and a bloated prompt makes the grading noisier.
    context = [
        {k: v for k, v in item.items() if k != "matched_text"} | {"excerpt": str(item.get("matched_text", ""))[:400]}
        for item in retrieved
    ]
    task = f"""QUESTION PUT TO THE ASSISTANT:
{prompt}

RETRIEVED CONTEXT THE ASSISTANT WAS GIVEN (the only source of truth about parks):
{json.dumps(context, indent=2, default=str)}

THE ASSISTANT'S ANSWER:
{answer}

RUBRIC - judge each criterion separately and return one ruling per criterion, in this order:
{chr(10).join(f"{i}. {c}" for i, c in enumerate(rubric, 1))}
"""
    verdict = model.with_structured_output(Verdict).invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=task)]
    )
    assert isinstance(verdict, Verdict)
    if len(verdict.criteria) != len(rubric):
        # A judge that skipped or invented a criterion has not graded what we asked, and silently
        # averaging over the wrong denominator would hide it.
        raise ValueError(
            f"judge returned {len(verdict.criteria)} rulings for {len(rubric)} criteria"
        )
    log.info("judge: passed=%s score=%s - %s", verdict.passed, verdict.score, verdict.reason)
    return verdict
