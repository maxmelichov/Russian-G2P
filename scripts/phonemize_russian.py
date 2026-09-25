"""Russian text -> IPA with the dictionary + rules engine (data/russian_phonemizer.py).

    python scripts/phonemize_russian.py --text "Солнце уже село за старый замок."
    python scripts/phonemize_russian.py --text "..." --report      # which tier stressed each word
    python scripts/phonemize_russian.py --csv in.csv --column text --out out.csv --workers 8

Needs ru_lexicon/ (python scripts/build_ru_lexicon.py). CSV mode adds an ``ipa``
column; each worker loads the lexicon once (~1.5 GB RAM), so --workers is bounded
by memory. The old RUAccent+RUPhon driver is scripts/phonemize_russian_legacy.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _init():
    from data import russian_phonemizer

    russian_phonemizer._lexicon()


def _one(text: str) -> str:
    from data.russian_phonemizer import phonemize_russian

    return phonemize_russian(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="append", help="sentence(s) to phonemize; repeatable")
    ap.add_argument("--report", action="store_true", help="print the stress tier of every word")
    ap.add_argument("--csv", help="input CSV")
    ap.add_argument("--column", default="text")
    ap.add_argument("--out", help="output CSV (default: <csv stem>_ipa.csv)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    from data.russian_phonemizer import last_report, phonemize_russian
    from data.text_vocab import unknown_symbols

    if a.text:
        for t in a.text:
            ipa = phonemize_russian(t)
            print(f"{t}\n  ipa: {ipa}")
            if a.report:
                for w, tier, p in last_report():
                    print(f"    {w:20s} {tier:22s} {p}")
            oov = unknown_symbols(ipa)
            if oov:
                print(f"  WARNING: symbols outside the vocab -> PAD: {oov}", file=sys.stderr)
        return 0
    if not a.csv:
        ap.error("give --text or --csv")
    with open(a.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [r[a.column] for r in rows]
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        ipa = list(pool.imap(_one, texts, chunksize=64))
    out = a.out or os.path.splitext(a.csv)[0] + "_ipa.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + (["ipa"] if "ipa" not in rows[0] else []))
        w.writeheader()
        for r, p in zip(rows, ipa):
            r["ipa"] = p
            w.writerow(r)
    print(f"{len(rows)} rows in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
