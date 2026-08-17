"""Text tables for the scripts. No plotting dependencies."""

from __future__ import annotations


def rule(title: str) -> None:
    print(f"\n{title}\n" + "─" * max(60, len(title)))


def table(rows: list[dict], cols: list[str] | None = None) -> None:
    if not rows:
        return
    cols = cols or list(rows[0])
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols))


def pm(mean: float, std: float, digits: int = 3) -> str:
    """"0.283 ± 0.007" - never report a mean without its spread."""
    return f"{mean:.{digits}f} ± {std:.{digits}f}"
