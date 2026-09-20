#!/usr/bin/env python3
"""Reset manual latency overrides to the empty automatic state."""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "manual_latency_overrides.json"
EMPTY = {
    "_guide": ("See 5_Plot/MANUAL_LATENCY_GUIDE.md. "
               "Put manual overrides under sl/cp/matched_recall."),
    "sl": {},
    "cp": {},
    "matched_recall": {},
}


def main():
    with open(PATH, "w", encoding="utf-8") as stream:
        json.dump(EMPTY, stream, indent=2, ensure_ascii=False)
    print(f"Reset {PATH}")


if __name__ == "__main__":
    main()