"""Sortable ULID-like ids: 48-bit millisecond timestamp + 80 bits of randomness,
Crockford base32, 26 chars. Lexicographic order matches creation order. Stdlib only."""

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_last_ms = 0
_last_rand = 0


def _encode(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_id() -> str:
    """Monotonic in-process: ids from the same millisecond still sort by call order."""
    global _last_ms, _last_rand
    ms = int(time.time() * 1000)
    if ms <= _last_ms:
        ms = _last_ms
        _last_rand += 1
        rand = _last_rand
    else:
        rand = int.from_bytes(os.urandom(10), "big")
    _last_ms, _last_rand = ms, rand
    return _encode(ms, 10) + _encode(rand & ((1 << 80) - 1), 16)
