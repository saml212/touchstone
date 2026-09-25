"""Cross-call minted ids: values one recorded call returned that a LATER recorded call references.

A create call (`save_passenger` -> `passenger_id=PAX-9`) mints an id a dependent call (`book_flight`
with `passenger_id=PAX-9`) then passes back. A naive simulator mints its OWN id on the create, so
the recorded id the dependent call carries is "not found" on replay. The generator needs to know
these ids and the inputs that produced them, so it can return the recorded id for the same input
and seed the entity — the referential integrity the recordings imply. Pure functions; fed into the
simulator prompt.
"""

from __future__ import annotations

import json


def _scalar_leaves(obj) -> set:
    """Hashable str/int leaves of a nested JSON value — the id candidates (bools excluded)."""
    if isinstance(obj, dict):
        return set().union(*(_scalar_leaves(v) for v in obj.values())) if obj else set()
    if isinstance(obj, (list, tuple)):
        return set().union(*(_scalar_leaves(v) for v in obj)) if obj else set()
    if isinstance(obj, bool):
        return set()
    if isinstance(obj, str) and obj:
        return {obj}
    if isinstance(obj, int):
        return {obj}
    return set()


def _later_arg_values(examples: list[dict], start: int) -> set:
    values: set = set()
    for other in examples[start:]:
        values |= _scalar_leaves(other.get("arguments"))
    return values


def minted_id_bindings(examples: list[dict]) -> list[dict]:
    """For each id a recorded call RETURNED that a later call passed as an argument: the creating
    call's tool + arguments + that id. `examples` are `{tool, arguments, returned}` in recorded
    order (the same list handed to the simulator prompt)."""
    bindings: list[dict] = []
    seen: set = set()
    for i, ex in enumerate(examples):
        shared = _scalar_leaves(ex.get("returned")) & _later_arg_values(examples, i + 1)
        for value in sorted(shared - seen, key=str):
            seen.add(value)
            bindings.append({"created_by": ex.get("tool"), "arguments": ex.get("arguments"),
                             "id": value})
    return bindings


_HEADER = (
    "## Cross-call ids — these ids were returned by an earlier recorded call and then passed to a "
    "LATER call, so they MUST resolve on replay. For each: when the creating call is replayed with "
    "the SAME inputs, return this EXACT id (derive the minted id deterministically from the "
    "request, do not mint a random one), AND seed the entity in seed.json so a dependent call "
    "replayed in isolation still finds it. Mint a fresh id only for inputs not listed here.")


def minted_ids_text(examples: list[dict]) -> str:
    """The cross-call-id section for the simulator prompt, or a short note when there are none."""
    bindings = minted_id_bindings(examples)
    if not bindings:
        return "(no cross-call ids detected)"
    return _HEADER + "\n" + json.dumps(bindings, indent=2, ensure_ascii=False)
