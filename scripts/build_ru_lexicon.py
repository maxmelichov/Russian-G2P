"""Build ru_lexicon/: the stress data and context models data/russian_phonemizer.py reads.

    python scripts/build_ru_lexicon.py                       # download from the Hub + build
    python scripts/build_ru_lexicon.py --src DIR             # from an installed ruaccent package dir
    python scripts/build_ru_lexicon.py --labels meta_ipa.csv # add corpus priors (optional)

Inputs are RUAccent's dictionaries and two of its ONNX models (HF repo
``ruaccent/accentuator``). Everything is read with the standard library,
``huggingface_hub`` (download only) and nothing else -- no ruaccent, no
transformers. Outputs (``ru_lexicon/``, gitignored: ~375 MB of third-party data):

* ``accents.tsv.gz``       word -> index of the stressed vowel (0-based, among vowels);
                           3.19 M Zaliznyak-derived word forms
* ``yo.tsv.gz``            е-spelling -> ё-spelling, for words that are unambiguous
* ``yo_homographs.tsv.gz`` е-spelling -> ё-spelling where both words exist (все/всё);
                           the ``yo_model`` tagger decides in context
* ``homographs.tsv.gz``    word -> "i1,i2,...", candidate stress indices, most likely first
* ``homograph_ctx.tsv.gz`` "p|prev|word" / "n|word|next" -> stress index (corpus prior)
* ``suffix_stress.tsv.gz`` word ending -> stress position counted from the END, the
                           out-of-dictionary fallback (majority over the dictionary)
* ``train_words.tsv.gz``   word -> stress index observed in a labelled corpus
* ``omograph/``, ``yo_model/`` RUAccent's homograph classifier (turbo3.1) and
                           ё-homograph tagger, run with onnxruntime + tokenizers

``--labels`` takes a CSV with ``text`` (plain lower-case words) and ``ipa``
(IPA with ˈ, one token per word) columns -- e.g. a corpus phonemized by the
legacy RUAccent+RUPhon driver. It only orders homograph candidates and fills
the two corpus-prior files, which are used when the context models are
missing or a word is not in the dictionary; without it those files are empty.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import shutil
import sys

VOWELS = "аеёиоуыэюя"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "ru_lexicon")
FILES = ["accents.json.gz", "yo_words.json.gz", "omographs.json.gz", "yo_homographs.json.gz"]
MODELS = {"omograph": "nn/nn_omograph/turbo3.1", "yo_model": "nn/nn_yo_homograph_resolver"}


def stress_index(accented: str) -> int | None:
    """'мол+око' -> 1 (the stressed vowel is the 2nd vowel)."""
    if "+" not in accented:
        return None
    return sum(1 for c in accented[: accented.index("+")] if c in VOWELS)


def label_stress_index(ipa: str) -> int | None:
    v, marked = 0, False
    for ch in ipa:
        if ch == "ˈ":
            marked = True
            continue
        if ch in "aeiouɨɪəɐʊæɵʉɛ":
            if marked:
                return v
            v += 1
    return None


def fetch(src_dir: str | None, out: str) -> str:
    """-> a directory in RUAccent's layout (dictionary/, nn/)."""
    if src_dir:
        return src_dir
    from huggingface_hub import hf_hub_download, list_repo_files

    cache = os.path.join(out, "_download")
    files = list_repo_files("ruaccent/accentuator")
    want = [f"dictionary/{f}" for f in FILES]
    want += [f for f in files if any(f.startswith(m + "/") for m in MODELS.values())]
    for f in want:
        hf_hub_download(repo_id="ruaccent/accentuator", filename=f, local_dir=cache)
    return cache


def write_tsv(path: str, rows) -> int:
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="an installed ruaccent package dir (holding dictionary/ and nn/)")
    ap.add_argument("--labels", default=None, help="optional CSV with text,ipa columns (corpus priors)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = fetch(a.src, a.out)

    for name, rel in MODELS.items():
        s = os.path.join(root, rel)
        if not os.path.isdir(s):
            print(f"[warn] {rel} not found under {root}; homographs fall back to the dictionary order")
            continue
        d = os.path.join(a.out, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(s, d)
        print("model", name, "->", d)

    src = os.path.join(root, "dictionary")
    acc = json.load(gzip.open(os.path.join(src, "accents.json.gz")))
    yo = json.load(gzip.open(os.path.join(src, "yo_words.json.gz")))
    om = json.load(gzip.open(os.path.join(src, "omographs.json.gz")))
    yo_h = json.load(gzip.open(os.path.join(src, "yo_homographs.json.gz")))

    rows = [(w, stress_index(v)) for w, v in acc.items() if stress_index(v) is not None]
    print("accents", write_tsv(os.path.join(a.out, "accents.tsv.gz"), sorted(rows)))
    print("yo", write_tsv(os.path.join(a.out, "yo.tsv.gz"), sorted((k, v) for k, v in yo.items() if k not in yo_h)))
    print("yo homographs", write_tsv(os.path.join(a.out, "yo_homographs.tsv.gz"), sorted(yo_h.items())))

    lab: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ctx: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    if a.labels:
        csv.field_size_limit(sys.maxsize)
        with open(a.labels, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tw, iw = str(r["text"]).split(), str(r["ipa"]).split()
                if len(tw) != len(iw):
                    continue
                for j, (w, p) in enumerate(zip(tw, iw)):
                    k = label_stress_index(p)
                    if k is None or sum(c in VOWELS for c in w) < 2:
                        continue
                    lab[w][k] += 1
                    if w in om:
                        ctx[f"p|{tw[j - 1] if j else '<s>'}|{w}"][k] += 1
                        ctx[f"n|{w}|{tw[j + 1] if j + 1 < len(tw) else '</s>'}"][k] += 1

    hom_rows = []
    for w, variants in om.items():
        idx = list(dict.fromkeys(i for i in (stress_index(v) for v in variants) if i is not None))
        c = lab.get(w, collections.Counter())
        idx.sort(key=lambda k: -c.get(k, 0))
        if len(idx) > 1:
            hom_rows.append((w, ",".join(map(str, idx))))
    print("homographs", write_tsv(os.path.join(a.out, "homographs.tsv.gz"), sorted(hom_rows)))
    ctx_rows = [(k, c.most_common(1)[0][0]) for k, c in ctx.items()
                if sum(c.values()) >= 2 and c.most_common(1)[0][1] / sum(c.values()) >= 0.8]
    print("homograph contexts", write_tsv(os.path.join(a.out, "homograph_ctx.tsv.gz"), sorted(ctx_rows)))

    suf: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for w, k in rows:
        nv = sum(c in VOWELS for c in w)
        if nv < 2:
            continue
        for L in (3, 4, 5, 6):
            if len(w) > L:
                suf[w[-L:]][nv - 1 - k] += 1
    suf_rows = [(s, c.most_common(1)[0][0]) for s, c in suf.items()
                if sum(c.values()) >= 5 and c.most_common(1)[0][1] / sum(c.values()) >= 0.6]
    print("suffixes", write_tsv(os.path.join(a.out, "suffix_stress.tsv.gz"), sorted(suf_rows)))
    print("train words", write_tsv(os.path.join(a.out, "train_words.tsv.gz"),
                                   sorted((w, c.most_common(1)[0][0]) for w, c in lab.items())))
    shutil.rmtree(os.path.join(a.out, "_download"), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
