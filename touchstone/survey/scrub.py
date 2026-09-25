"""Replace PII with stable fakes before any recording enters a generated file.

Covers emails, card-like numbers, phone numbers, and full names named in `[survey] names`. The
same input maps to the same fake within one `Scrubber` (so seeds and examples stay consistent),
and the mapping is held in memory only — it is never written to disk. Pure and deterministic.
"""

from __future__ import annotations

import re
from collections import defaultdict

# One combined, left-to-right pass so a replacement is never re-scanned by a later pattern (a card
# fake like "4000-0000-0000-0001" would otherwise be re-matched as a phone). Alternation order is
# the priority: email, then card (13–19 digits), then phone. Phone allows a leading `(area)` and
# runs of separators so `(415) 555-0132` is caught; the 10-digit floor keeps short ids and ISO
# dates (8 digits) out — the simpler choice over full locale parsing.
_PII = re.compile(
    r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"|(?P<card>\b(?:\d[ -]?){12,18}\d\b)"
    r"|(?P<phone>\+?\(?\d(?:[ ()\-]{0,2}\d){9,14})"
)
_TEMPLATES = {
    "email": "person{n}@example.invalid",
    "card": "4000-0000-0000-{n:04d}",
    "phone": "5550100{n:04d}",
}


class Scrubber:
    def __init__(self, names: list[str] | None = None) -> None:
        self._fakes: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._names = [n for n in (names or []) if n and n.strip()]

    def _fake(self, category: str, key: str, template: str) -> str:
        mkey = (category, key)
        if mkey not in self._fakes:
            self._counts[category] += 1
            self._fakes[mkey] = template.format(n=self._counts[category])
        return self._fakes[mkey]

    def _replace(self, m: re.Match) -> str:
        category = m.lastgroup
        return self._fake(category, m.group(0), _TEMPLATES[category])

    def _sub_names(self, text: str) -> str:
        for name in self._names:
            pat = re.compile(re.escape(name), re.IGNORECASE)
            text = pat.sub(lambda m, k=name: self._fake("name", k.lower(), "Person {n}"), text)
        return text

    def text(self, value: str) -> str:
        return self._sub_names(_PII.sub(self._replace, value))

    def scrub(self, value):
        """Return a scrubbed copy of any JSON-like value (str / dict / list / scalar)."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        return value
