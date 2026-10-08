"""Conversation requirements are separate from the generator's JSON contract."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Requirement(Strict):
    id: str = Field(pattern=r"^R[0-9]+$")
    text: str = Field(min_length=1)
    status: Literal["confirmed", "proposed", "deferred"]


class Volume(Strict):
    entity: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    count: int = Field(gt=0)


class Brief(Strict):
    """User intent and approved assumptions, not an executable generator config."""

    name: str
    purpose: str
    requirements: list[Requirement]
    volumes: list[Volume]
    seed: int | None
    questions: list[str]
    assumptions: list[str]

    @model_validator(mode="after")
    def unique_ids(self):
        for values in ([r.id for r in self.requirements], [v.entity for v in self.volumes]):
            if len(values) != len(set(values)):
                raise ValueError("Requirement IDs and volume entity names must be unique.")
        return self


class Interview(Strict):
    reply: str
    brief: Brief


class Mapping(Strict):
    requirement_id: str
    explanation: str
    pointers: list[str]


class Design(Strict):
    """Candidate contract, traceable requirement coverage, and categorized feedback."""

    summary: str
    contract_json: str
    coverage: list[Mapping]
    unresolved: list[str]
    warnings: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class Review(Strict):
    summary: str
    missing_requirements: list[str]
    concerns: list[str]
    warnings: list[str]


class RequirementFinding(Strict):
    """One advisory dataset verdict backed by IDs in the measured evidence packet."""

    requirement_id: str
    verdict: Literal["supported", "mismatch", "insufficient_evidence", "deferred"]
    evidence_ids: list[str]
    explanation: str
    suggested_action: str


class DatasetReview(Strict):
    summary: str
    requirements: list[RequirementFinding]
    limitations: list[str]


def blank_brief():
    return Brief(
        name="", purpose="", requirements=[], volumes=[], seed=None, questions=[], assumptions=[]
    ).model_dump()
