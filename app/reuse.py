"""Reuse of historical conflict resolutions across merge orders.

Given a *target* merge's currently open conflicts, a *source* merge's
conflict snapshot (taken when the source was created) and the source's
immutable final result, every open target conflict is matched against the
source conflict at the same RFC 6901 JSON Pointer:

  * no source conflict at that pointer            -> skipped
  * base/left/right differ after normalisation    -> skipped
  * the ``missing`` sides differ                  -> skipped
  * otherwise the value is read from the source's
    final result; an absent key means the source
    chose a deletion, honoured the same way here   -> applied

Normalisation distinguishes a genuinely missing key (a side named in the
conflict's ``missing`` list) from an explicit JSON ``null``: a missing
side never matches a present ``null``. JSON equality is otherwise strict
about types as well — unlike Python's ``==``, ``true`` is not equal to
1. Everything here is pure and in-memory; the caller decides whether the
outcome is persisted.
"""

from __future__ import annotations

from typing import Any

from .merge import MISSING, parse_pointer

REASON_NO_SOURCE_CONFLICT = "no_source_conflict"
REASON_SHAPE_MISMATCH = "shape_mismatch"

ACTION_SET = "set"
ACTION_DELETE = "delete"

_SIDES = ("base", "left", "right")


def json_equal(a: Any, b: Any) -> bool:
    """Structural equality over JSON values.

    Unlike Python ``==``: a missing key (``MISSING``) only equals a missing
    key, JSON ``null`` only equals ``null``, booleans never equal numbers
    (``True`` is not ``1``), while ints and floats compare as the single JSON
    number type (``1 == 1.0``).
    """
    if a is MISSING or b is MISSING:
        return a is MISSING and b is MISSING
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(json_equal(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            json_equal(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _side_value(conflict: dict, side: str) -> Any:
    """Return a conflict side with missing-key semantics.

    A side named in ``missing`` normalises to ``MISSING``; every other side
    is returned as-is, including an explicit JSON ``null``.
    """
    if side in conflict.get("missing", []):
        return MISSING
    return conflict[side]


def shape_differences(target: dict, source: dict) -> list[str]:
    """Names of the normalised fields in which two conflicts differ.

    Compares ``missing`` (as a set of side names) plus ``base``, ``left``
    and ``right`` with missing-vs-null-aware equality. An empty list means
    the shapes are identical and the source resolution is reusable.
    """
    differences: list[str] = []
    if sorted(target.get("missing", [])) != sorted(source.get("missing", [])):
        differences.append("missing")
    for side in _SIDES:
        if not json_equal(_side_value(target, side), _side_value(source, side)):
            differences.append(side)
    return differences


def lookup_final(result: Any, tokens: list[str]) -> Any:
    """Read the source's resolved value at ``tokens``; ``MISSING`` if absent.

    A missing key anywhere on the path (or an index past an array end)
    means the source result no longer carries the path: the resolution was
    a deletion. The document root itself is returned for an empty pointer.
    """
    node = result
    for token in tokens:
        if isinstance(node, dict):
            if token not in node:
                return MISSING
            node = node[token]
        elif isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return MISSING
        else:
            return MISSING
    return node


def plan_reuse(
    target_conflicts: list[dict],
    source_conflicts: list[dict],
    source_result: Any,
) -> tuple[list[dict], list[dict]]:
    """Match every open target conflict against the source snapshot.

    Returns (applied, skipped), both ordered by pointer. Each applied item
    is ``{"pointer", "action": "set", "value"}`` reading the value from the
    source's final result, or ``{"pointer", "action": "delete"}`` when the
    source result no longer carries the path. Each skipped item is
    ``{"pointer", "reason"}`` with ``differences`` naming the mismatching
    fields for a shape mismatch.
    """
    source_by_pointer = {c["pointer"]: c for c in source_conflicts}
    applied: list[dict] = []
    skipped: list[dict] = []
    for target in sorted(target_conflicts, key=lambda c: c["pointer"]):
        pointer = target["pointer"]
        source = source_by_pointer.get(pointer)
        if source is None:
            skipped.append(
                {"pointer": pointer, "reason": REASON_NO_SOURCE_CONFLICT}
            )
            continue
        differences = shape_differences(target, source)
        if differences:
            skipped.append(
                {
                    "pointer": pointer,
                    "reason": REASON_SHAPE_MISMATCH,
                    "differences": differences,
                }
            )
            continue
        chosen = lookup_final(source_result, parse_pointer(pointer))
        if chosen is MISSING:
            applied.append({"pointer": pointer, "action": ACTION_DELETE})
        else:
            applied.append(
                {"pointer": pointer, "action": ACTION_SET, "value": chosen}
            )
    return applied, skipped
