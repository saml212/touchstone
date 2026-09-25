"""Mask volatile fields before comparing a replayed response with the recorded one.

A fresh run mints new ids, tickets, timestamps and tokens, so comparing raw values would count every
one of those as a mismatch. `_compare` masks volatile keys (by name) and ISO timestamps (by value)
on both sides first, so only a real difference in the stable fields fails. Pure functions, shared by
fidelity scoring and the task criteria helpers.
"""

from __future__ import annotations

import re

_VOLATILE_EXACT = {"id", "ticket", "timestamp", "ts", "token"}
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _is_volatile(key: str) -> bool:
    k = key.lower()
    return (k in _VOLATILE_EXACT or k.endswith("_id")
            or k.startswith("created") or k.startswith("updated"))


def _mask(value, masked: set):
    if isinstance(value, dict):
        return {k: _mask_field(k, v, masked) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v, masked) for v in value]
    if isinstance(value, str) and _ISO.search(value):
        masked.add("<iso-timestamp>")
        return "<ts>"
    return value


def _mask_field(key: str, value, masked: set):
    if _is_volatile(key):
        masked.add(key)
        return "<masked>"
    return _mask(value, masked)


def _compare(expected, got) -> tuple[bool, set]:
    masked: set = set()
    return _mask(expected, masked) == _mask(got, masked), masked


def masked_equal(expected, got) -> bool:
    """True when `got` matches `expected` after masking volatile fields (ids, timestamps, …)."""
    return _compare(expected, got)[0]
