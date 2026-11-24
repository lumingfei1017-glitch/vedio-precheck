from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BatchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class SuggestionUpdate(BaseModel):
    after_text: str | None = None
    method: str | None = None
    abc_class: str | None = None
    status: str | None = None


class RefreshRequest(BaseModel):
    constraint: str | None = None
    instruction: str | None = None


class ConfirmRequest(BaseModel):
    accepted: bool = True


class FeedbackCreate(BaseModel):
    action: str = Field(default="false_positive", pattern="^(false_positive|correct)$")
    reason: str | None = Field(default=None, max_length=500)


class ShotReplaceRequest(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=200)
    route: Literal["wan", "tts"] = "wan"
    dry_run: bool = True


class SuggestionOut(BaseModel):
    id: str
    finding_id: str
    before_text: str
    after_text: str
    method: str
    abc_class: str
    estimated_duration_ms: int | None
    original_duration_ms: int | None
    duration_warning: bool
    confidence: float
    status: str


class FindingOut(BaseModel):
    id: str
    rule_id: str
    title: str
    severity: str
    category: str
    modality: list[str]
    quote: str
    rationale: str
    start_ms: int | None
    end_ms: int | None
    confidence: float
    needs_manual_time: bool
    source: str = "rule"
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    suggestion: SuggestionOut | None = None


class MaterialOut(BaseModel):
    id: str
    batch_id: str
    filename: str
    category: str
    has_face: bool | None
    status: str
    duration_ms: int
    created_at: str


class MaterialReport(BaseModel):
    material: MaterialOut
    summary: dict[str, Any]
    findings: list[FindingOut]
    recheck: dict[str, Any] | None = None


class DryRunResponse(BaseModel):
    material_id: str
    dry_run: bool
    plan: list[dict[str, Any]]
    estimated_duration_ms: int
    estimated_cost: float
    cost_note: str
    warnings: list[str]
