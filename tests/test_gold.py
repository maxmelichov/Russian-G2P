"""Russian G2P gold-set test: data/russian_phonemizer.py against tests/gold.tsv.

    python tests/test_gold.py          # prints every miss, exits 1 on any
    python -m pytest tests/test_gold.py

gold.tsv (tab-separated, '#' lines are comments):

    word      <word>      v1|v2|...                 <source: Wiktionary URL + the raw IPA>
    sentence  <sentence>  w1a|w1b w2 w3a|w3b ...    <note>

Gold IPA is en.wiktionary's ru-pron standard pronunciation (the transcription
RUPhon imitates), converted to the label
notation: tie-bar affricates -> ʦ ʧ ʣ ʤ, ˈ moved from the syllable onset to the
vowel (to the j of an iotated vowel), optional segments ⁽ʲ⁾ (ː) (j) expanded to
both variants, dialect lines dropped. Monosyllables follow the label convention
(no ˈ except MONO_STRESSED and ё words) and also accept the unreduced citation
vowel and final devoicing, which is how the labels write a clitic in isolation.
A sentence's gold is its words' gold in context: homograph stress fixed by
meaning. Punctuation is not scored here; the out-of-vocabulary check below
covers it.

Requires ru_lexicon/ (scripts/build_ru_lexicon.py).
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold.tsv")
_PUNCT = re.compile(r"[.,!?;:]")


def load_gold(path: str = GOLD) -> list[tuple[str, str, list[list[str]]]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            kind, text, gold, *_ = line.rstrip("\n").split("\t")
            slots = [slot.split("|") for slot in gold.split(" ")] if kind == "sentence" else [gold.split("|")]
            rows.append((kind, text, slots))
    return rows


def check_row(kind: str, text: str, slots: list[list[str]]) -> tuple[bool, str]:
    from data.russian_phonemizer import phonemize_russian

    out = _PUNCT.sub("", phonemize_russian(text)).split()
    if len(out) != len(slots):
        return False, f"{len(out)} words out, {len(slots)} in gold: {' '.join(out)}"
    bad = [(o, s) for o, s in zip(out, slots) if o not in s]
    if bad:
        return False, "; ".join(f"{o} not in {'|'.join(s)}" for o, s in bad)
    return True, ""


def run() -> tuple[int, int, list[str]]:
    from data.text_vocab import CHAR_TO_ID
    from data.russian_phonemizer import phonemize_russian

    rows = load_gold()
    fails = []
    for kind, text, slots in rows:
        ok, why = check_row(kind, text, slots)
        if not ok:
            fails.append(f"[{kind}] {text}: {why}")
        oov = {c for c in phonemize_russian(text) if c not in CHAR_TO_ID}
        if oov:
            fails.append(f"[{kind}] {text}: characters outside CHAR_TO_ID (-> PAD): {sorted(oov)}")
    return len(rows), len(fails), fails


def test_gold_set():
    n, nf, fails = run()
    assert nf == 0, f"{nf}/{n} gold rows fail:\n" + "\n".join(fails)


def test_digits_expand():
    from data.russian_phonemizer import phonemize_russian

    assert not re.search(r"\d", phonemize_russian("В 1945 году было 25 человек, 3,5 %."))


def test_raw_cyrillic_never_reaches_the_vocab():
    from data.russian_phonemizer import phonemize_russian

    assert not re.search("[а-яё]", phonemize_russian("Привет, как дела? — «Хорошо»… (да)"), re.I)


if __name__ == "__main__":
    n, nf, fails = run()
    for f in fails:
        print(f)
    print(f"{n - nf}/{n} gold rows pass")
    sys.exit(1 if nf else 0)
