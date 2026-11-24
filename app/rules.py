from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .config import GENERATED_RULES_PATH, PROMPT_HINTS_PATH, RULES_PATH

SEVERITY_RANK = {"blocking": 3, "suggest": 2, "info": 1}
SENTENCE_BOUNDARY = "。！？!?；;，,\n"


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    category: str
    severity: str
    action: str
    default_method: str
    modality: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]
    negative_patterns: tuple[re.Pattern[str], ...]
    why: str
    replacements: tuple[tuple[re.Pattern[str], str], ...]
    fallback: str
    required_evidence: str


@dataclass(frozen=True)
class RuleMatch:
    rule: Rule
    match_text: str
    start: int
    end: int
    quote: str


def _compile(values: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(value, re.IGNORECASE) for value in values)


def _rule_items() -> list[dict]:
    items: list[dict] = []
    for path in (RULES_PATH, GENERATED_RULES_PATH):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items.extend(payload.get("rules", []))
    return items


@lru_cache(maxsize=1)
def load_rules() -> tuple[Rule, ...]:
    rules: list[Rule] = []
    for item in _rule_items():
        rewrite = item.get("rewrite", {})
        replacements = tuple(
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in rewrite.get("replacements", [])
        )
        rules.append(
            Rule(
                id=item["id"],
                title=item["title"],
                category=item["category"],
                severity=item["severity"],
                action=item["action"],
                default_method=item["default_method"],
                modality=tuple(item.get("modality", [])),
                patterns=_compile(item.get("patterns", [])),
                negative_patterns=_compile(item.get("negative_patterns", [])),
                why=item.get("why", ""),
                replacements=replacements,
                fallback=rewrite.get("fallback", ""),
                required_evidence=item.get("required_evidence", ""),
            )
        )
    return tuple(rules)


def rules_by_id() -> dict[str, Rule]:
    return {rule.id: rule for rule in load_rules()}


def rule_version() -> str:
    versions = []
    for path in (RULES_PATH, GENERATED_RULES_PATH):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        versions.append(str(payload.get("version", "unknown")))
    return "+".join(versions) if versions else "unknown"


def prompt_hints() -> list[str]:
    """从判例挖出来、但不足以做阻断规则的通用表述。"""
    if not PROMPT_HINTS_PATH.is_file():
        return []
    payload = json.loads(PROMPT_HINTS_PATH.read_text(encoding="utf-8"))
    return [str(item) for item in payload.get("hints", [])]


def sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(ch, 0, start) for ch in SENTENCE_BOUNDARY) + 1
    right_candidates = [text.find(ch, end) for ch in SENTENCE_BOUNDARY if text.find(ch, end) != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    sentence = text[left:right].strip().strip(" \t\r\n，,。！？!?；;")
    return sentence or text[start:end].strip()


def scan(text: str, rules: Iterable[Rule] | None = None) -> list[RuleMatch]:
    """Return word-level candidates. Candidates are not final verdicts."""
    if not text or not text.strip():
        return []

    active_rules = tuple(rules) if rules is not None else load_rules()
    matches: list[RuleMatch] = []
    seen: set[tuple[int, int, str]] = set()

    for rule in active_rules:
        for pattern in rule.patterns:
            for hit in pattern.finditer(text):
                key = (hit.start(), hit.end(), rule.id)
                if key in seen:
                    continue
                quote = sentence_around(text, hit.start(), hit.end())
                if any(negative.search(quote) for negative in rule.negative_patterns):
                    continue
                seen.add(key)
                matches.append(
                    RuleMatch(
                        rule=rule,
                        match_text=hit.group(0),
                        start=hit.start(),
                        end=hit.end(),
                        quote=quote,
                    )
                )

    # Within one rule, keep the longest span (e.g. "显瘦十几斤" over "瘦十几斤").
    filtered: list[RuleMatch] = []
    for match in matches:
        contained = any(
            other.rule.id == match.rule.id
            and other.start <= match.start
            and other.end >= match.end
            and (other.start, other.end) != (match.start, match.end)
            for other in matches
        )
        if not contained:
            filtered.append(match)

    # 判例自动挖出来的规则（MK-*）不能和手写规则重复报同一句。
    deduped: list[RuleMatch] = []
    for match in filtered:
        if match.rule.id.startswith("MK-") and any(
            other.rule.id != match.rule.id
            and not other.rule.id.startswith("MK-")
            and other.start <= match.start
            and other.end >= match.end
            for other in filtered
        ):
            continue
        deduped.append(match)
    filtered = deduped

    filtered.sort(key=lambda item: (item.start, item.end, -SEVERITY_RANK.get(item.rule.severity, 0)))
    # One finding per rule per clause is enough for the MVP report.
    by_rule_quote: dict[tuple[str, str], RuleMatch] = {}
    for match in filtered:
        key = (match.rule.id, match.quote)
        existing = by_rule_quote.get(key)
        if existing is None or len(match.match_text) > len(existing.match_text):
            by_rule_quote[key] = match
    return sorted(by_rule_quote.values(), key=lambda item: (item.start, item.end))
