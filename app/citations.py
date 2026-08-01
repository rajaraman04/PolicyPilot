"""Citation parsing utilities shared by the app pipeline and the eval harness.

Lives in app/ so app/verifier.py can reuse it without app/ importing from eval/.
eval/metrics.py re-exports these names for backward compatibility.
"""

from __future__ import annotations

import re

# Matches the citation format our prompt asks for: (nist_csf.pdf, p.8)
# Tolerates "p.8", "p 8", "pp. 8".
CITATION_RE = re.compile(r"\(\s*([\w\-.]+\.pdf)\s*,\s*pp?\.?\s*(\d+)\s*\)", re.IGNORECASE)


def parse_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (document, page) pairs cited inline in an answer."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(answer)]


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking on the '.' inside '(file.pdf, p.8)'.

    Citations are masked before splitting, then restored.
    """
    spans = list(CITATION_RE.finditer(text))
    masked = text
    for i, m in enumerate(reversed(spans)):
        idx = len(spans) - 1 - i
        masked = masked[: m.start()] + f"\x00C{idx}\x00" + masked[m.end() :]

    parts = re.split(r"(?<=[.!?])\s+", masked)

    out = []
    for part in parts:
        restored = part
        for idx, m in enumerate(spans):
            restored = restored.replace(f"\x00C{idx}\x00", m.group(0))
        if restored.strip():
            out.append(restored.strip())
    return out
