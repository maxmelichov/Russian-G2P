"""Russian text -> narrow IPA on the 256-token vocab. No transformers, no RUPhon.

Pipeline, per sentence:

1. **Text normalization** to what a TTS label is made from: numbers, years,
   dates, decimals, ``%`` and currency expanded to words; dashes, quotes and
   brackets dropped; hyphenated words split; ``. , ! ? : ;`` kept. Latin-script
   words go to eSpeak ``en-us`` through ``phonemizer`` when it is installed
   (optional; without it they are left as Latin letters).
2. **ё and stress** from ``ru_lexicon/`` (``scripts/build_ru_lexicon.py``):
   RUAccent's 3.19 M-form dictionary; ё-homographs (все/всё ...) by RUAccent's
   ё-tagger and stress homographs (замок, дела ...) by its turbo3.1 classifier,
   both ONNX graphs run here with onnxruntime + tokenizers. If the models are
   absent, a corpus prior decides. Words absent from the dictionary: corpus
   lexicon, then the legacy RUAccent+RUPhon engine if importable, else a
   suffix-analogy stress guess. ``last_report()`` says which tier each word used.
3. **Word -> IPA** by ``data.russian_g2p_rules.word_to_ipa`` (Moscow-norm rules,
   one word at a time), already in the vocab's symbols (ʦ ʧ ʤ ligatures, ˈ).
4. ``data.text_vocab.normalize_text``.
"""

from __future__ import annotations

import gzip
import os
import re
import unicodedata

from data.russian_g2p_rules import word_to_ipa

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_DIR = os.environ.get("RU_LEXICON_DIR", os.path.join(_REPO, "ru_lexicon"))
VOWELS = "аеёиоуыэюя"
_KEEP_PUNCT = ".,!?:;"

_LEX: dict | None = None
_REPORT: list[tuple[str, str, str]] = []


def _read_tsv(name: str) -> dict[str, str]:
    path = os.path.join(LEXICON_DIR, name)
    out: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("\t")
            out[k] = v
    return out


def _lexicon() -> dict:
    global _LEX
    if _LEX is None:
        if not os.path.isfile(os.path.join(LEXICON_DIR, "accents.tsv.gz")):
            raise FileNotFoundError(
                f"Russian stress lexicon missing from {LEXICON_DIR}. Build it once with\n"
                "  python scripts/build_ru_lexicon.py\n"
                "(or point RU_LEXICON_DIR at an existing copy). There is no silent fallback: "
                "without stress the output would be wrong on most words."
            )
        _LEX = {
            "acc": {k: int(v) for k, v in _read_tsv("accents.tsv.gz").items()},
            "yo": _read_tsv("yo.tsv.gz"),
            "hom": {k: [int(x) for x in v.split(",")] for k, v in _read_tsv("homographs.tsv.gz").items()},
            "suf": {k: int(v) for k, v in _read_tsv("suffix_stress.tsv.gz").items()},
            "train": {k: int(v) for k, v in _read_tsv("train_words.tsv.gz").items()},
            "ctx": {k: int(v) for k, v in _read_tsv("homograph_ctx.tsv.gz").items()},
            "yo_hom": _read_tsv("yo_homographs.tsv.gz"),
        }
    return _LEX


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_UNITS_F = {1: "одна", 2: "две"}
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
          "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
         "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
             "восемьсот", "девятьсот"]
_SCALES = [("тысяча", "тысячи", "тысяч", True), ("миллион", "миллиона", "миллионов", False),
           ("миллиард", "миллиарда", "миллиардов", False)]
# ordinal stems (masculine nominative) for the LAST word of a number
_ORD = {
    "ноль": "нулев", "один": "перв", "два": "втор", "три": "трет", "четыре": "четвёрт",
    "пять": "пят", "шесть": "шест", "семь": "седьм", "восемь": "восьм", "девять": "девят",
    "десять": "десят", "одиннадцать": "одиннадцат", "двенадцать": "двенадцат",
    "тринадцать": "тринадцат", "четырнадцать": "четырнадцат", "пятнадцать": "пятнадцат",
    "шестнадцать": "шестнадцат", "семнадцать": "семнадцат", "восемнадцать": "восемнадцат",
    "девятнадцать": "девятнадцат", "двадцать": "двадцат", "тридцать": "тридцат",
    "сорок": "сороков", "пятьдесят": "пятидесят", "шестьдесят": "шестидесят",
    "семьдесят": "семидесят", "восемьдесят": "восьмидесят", "девяносто": "девяност",
    "сто": "сот", "двести": "двухсот", "триста": "трёхсот", "четыреста": "четырёхсот",
    "пятьсот": "пятисот", "шестьсот": "шестисот", "семьсот": "семисот", "восемьсот": "восьмисот",
    "девятьсот": "девятисот", "тысяча": "тысячн", "миллион": "миллионн",
}


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def _triple(n: int, feminine: bool = False) -> list[str]:
    out = []
    h, r = divmod(n, 100)
    if h:
        out.append(_HUNDREDS[h])
    if 10 <= r < 20:
        out.append(_TEENS[r - 10])
    else:
        t, u = divmod(r, 10)
        if t:
            out.append(_TENS[t])
        if u:
            out.append(_UNITS_F[u] if feminine and u in _UNITS_F else _UNITS[u])
    return out


def _number_to_words(n: int, feminine: bool = False) -> list[str]:
    if n == 0:
        return ["ноль"]
    words: list[str] = []
    parts = []
    while n:
        n, r = divmod(n, 1000)
        parts.append(r)
    for scale in range(len(parts) - 1, -1, -1):
        r = parts[scale]
        if not r:
            continue
        if scale == 0:
            words += _triple(r, feminine)
        else:
            one, few, many, fem = _SCALES[scale - 1]
            tri = _triple(r, fem)
            if not (scale == 1 and r == 1):
                words += tri
            words.append(_plural(r, (one, few, many)))
    return words


_GOD_CASE = {"год": "nom", "года": "gen", "году": "prep", "годом": "ins", "годов": "gen", "годах": "prep",
             "годам": "dat", "годами": "ins"}
_ORD_END_HARD = {"nom": "ый", "gen": "ого", "dat": "ому", "prep": "ом", "ins": "ым", "neut": "ое"}
_ORD_END_STRESSED = dict(_ORD_END_HARD, nom="ой")
_ORD_END_SOFT = {"nom": "ий", "gen": "ьего", "dat": "ьему", "prep": "ьем", "ins": "ьим", "neut": "ье"}
# "9 мая" -> девятое мая (a date is a neuter ordinal: число)
_MONTHS_GEN = {"января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
               "сентября", "октября", "ноября", "декабря"}


def _ordinal(words: list[str], case: str) -> list[str]:
    """Cardinal words -> ordinal (only the last word changes), in ``case``."""
    if words[-2:] == ["две", "тысячи"] and len(words) == 2:
        return ["двухтысячн" + _ORD_END_HARD[case]]
    stem = _ORD.get(words[-1])
    if stem is None:
        return words
    if stem == "трет":
        end = _ORD_END_SOFT[case]
    elif stem in ("втор", "шест", "седьм", "восьм", "сороков"):
        end = _ORD_END_STRESSED[case]
    else:
        end = _ORD_END_HARD[case]
    return words[:-1] + [stem + end]


def _expand_numbers(text: str) -> str:
    def repl(m: re.Match) -> str:
        num = m.group(1).replace(" ", "").replace(" ", "")
        frac = m.group(2)
        after = text[m.end():]
        nxt = re.match(r"\s*([а-яёА-ЯЁ]+)", after)
        nxt_w = nxt.group(1).lower() if nxt else ""
        if frac:
            a = _number_to_words(int(num), feminine=True)
            b = _number_to_words(int(frac), feminine=True)
            den = {1: ("десятая", "десятых", "десятых"), 2: ("сотая", "сотых", "сотых"),
                   3: ("тысячная", "тысячных", "тысячных")}.get(len(frac))
            if den is None:
                return " ".join(a + ["запятая"] + b)
            return " ".join(a + [_plural(int(num), ("целая", "целых", "целых"))] + b + [_plural(int(frac), den)])
        n = int(num)
        if nxt_w in _GOD_CASE and 1000 <= n <= 2100:
            return " ".join(_ordinal(_number_to_words(n), _GOD_CASE[nxt_w]))
        if nxt_w in _MONTHS_GEN and 1 <= n <= 31:
            return " ".join(_ordinal(_number_to_words(n), "neut"))
        feminine = nxt_w.endswith(("а", "я", "ь")) and nxt_w not in ("человека", "года", "раза", "дня", "рубля")
        return " ".join(_number_to_words(n, feminine=feminine and n % 10 in (1, 2) and n % 100 not in (11, 12)))

    text = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + " " + _plural(int(m.group(1)), ("процент", "процента", "процентов")), text)
    return re.sub(r"(\d[\d  ]*\d|\d)(?:[.,](\d+))?", repl, text)


_SIGNS = {"$": "долларов", "€": "евро", "₽": "рублей", "£": "фунтов", "₪": "шекелей", "№": "номер",
          "&": "и", "+": "плюс", "=": "равно", "°": "градусов"}


# ---------------------------------------------------------------------------
# RUAccent's context models, run with onnxruntime + tokenizers (no transformers)
# ---------------------------------------------------------------------------
class _ContextModels:
    """Homograph stress classifier and ё-homograph tagger from RUAccent.

    Both are ONNX graphs with a HF ``tokenizer.json``, so they run without
    transformers. Missing model directories are not an error: the
    corpus prior (``homograph_ctx``/``homographs``) and the все/всё rule take over,
    and ``last_report()`` says so ('homograph-prior', 'context').
    """

    def __init__(self, root: str):
        self.root = root
        self._om = self._yo = None
        self.om_ok = os.path.isfile(os.path.join(root, "omograph", "model.onnx"))
        self.yo_ok = os.path.isfile(os.path.join(root, "yo_model", "model.onnx"))

    @staticmethod
    def _load(d: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(os.path.join(d, "model.onnx"), so, providers=["CPUExecutionProvider"])
        return Tokenizer.from_file(os.path.join(d, "tokenizer.json")), sess

    def _run(self, tok_sess, enc):
        import numpy as np

        tok, sess = tok_sess
        names = {i.name for i in sess.get_inputs()}
        feeds = {"input_ids": np.array([enc.ids], dtype=np.int64),
                 "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        return sess.run(None, feeds)[0][0]

    def choose_stress(self, marked_text: str, hypotheses: list[str]) -> str:
        """marked_text has the word wrapped in <w></w>; hypotheses are '+'-forms."""
        import numpy as np

        if self._om is None:
            self._om = self._load(os.path.join(self.root, "omograph"))
        best, best_p = hypotheses[0], -1.0
        for h in hypotheses:
            o = self._run(self._om, self._om[0].encode(marked_text, h))
            p = np.exp(o - o.max())
            p = float(p[1] / p.sum())
            if p > best_p:
                best, best_p = h, p
        return best

    def yo_labels(self, sentence: str) -> dict[int, str]:
        """char offset of each word start -> 'YO' / 'NO_YO' (averaged over subwords)."""
        import json as _json
        import numpy as np

        if self._yo is None:
            self._yo = self._load(os.path.join(self.root, "yo_model"))
            with open(os.path.join(self.root, "yo_model", "config.json")) as f:
                self._yo_labels = {int(k): v for k, v in _json.load(f)["id2label"].items()}
        tok, _ = self._yo
        enc = tok.encode(sentence)
        logits = self._run(self._yo, enc)
        out: dict[int, list] = {}
        cur = None
        for i, (tid, (a, b)) in enumerate(zip(enc.ids, enc.offsets)):
            if enc.special_tokens_mask[i]:
                continue
            piece = enc.tokens[i]
            if piece.startswith("##") and cur is not None:
                out[cur].append(logits[i])
            else:
                cur = a
                out[cur] = [logits[i]]
        res = {}
        for a, ls in out.items():
            z = np.stack(ls)
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            pr = (e / e.sum(axis=-1, keepdims=True)).mean(axis=0)
            yo_id = next(k for k, v in self._yo_labels.items() if v == "YO")
            # 0.8, not argmax: at 0.68 the tagger turns "и сел у окна" into сёл
            res[a] = "YO" if pr[yo_id] >= _YO_THRESHOLD else "NO_YO"
        return res


_CTX: _ContextModels | None = None
_YO_THRESHOLD = 0.8
# ё-homographs whose ё reading is colloquial; the literary norm (Zaliznyak) is е
_YO_LITERARY_E = {"далеко", "недалеко"}


def _ctx() -> _ContextModels:
    global _CTX
    if _CTX is None:
        _CTX = _ContextModels(LEXICON_DIR)
    return _CTX


# ---------------------------------------------------------------------------
# stress / ё
# ---------------------------------------------------------------------------
_VSE_PLURAL_NEXT = {"они", "мы", "вы", "эти", "те", "люди", "свои", "мои", "наши", "ваши", "трое", "двое",
                    "остальные", "другие", "были", "стали", "эти", "такие", "вместе"}
_VSE_SINGULAR_NEXT = {"это", "таки", "равно", "что", "же", "ещё", "еще", "время", "было", "то", "ли",
                      "так", "кончено", "будет", "более", "дело", "одно", "остальное", "ближе", "кругом",
                      "знаю", "понятно", "хорошо", "в", "на", "не", "и", "у", "от", "по", "с", "как"}


def _vse(next_word: str | None) -> str:
    """все (all, plural) vs всё (everything): the orthography rarely writes ё."""
    if not next_word:
        return "всё"
    n = next_word
    if n in _VSE_SINGULAR_NEXT:
        return "всё"
    if n in _VSE_PLURAL_NEXT or n.endswith(("ые", "ие", "ия", "ы", "ют", "ят", "ли", "ны", "и")):
        return "все"
    return "всё"


def _suffix_stress(w: str, lex: dict) -> int | None:
    vs = [i for i, c in enumerate(w) if c in VOWELS]
    if len(vs) < 2:
        return 0 if vs else None
    for L in (6, 5, 4, 3):
        if len(w) > L and w[-L:] in lex["suf"]:
            k = len(vs) - 1 - lex["suf"][w[-L:]]
            if 0 <= k < len(vs):
                return k
    return max(0, len(vs) - 2)  # penultimate: the commonest single position


_ENGINE = None


def _engine_word(w: str) -> str | None:
    """The legacy engine (RUAccent + RUPhon, data/russian_g2p.py), used only for
    out-of-dictionary words and only when both packages are importable
    (requirements-legacy.txt)."""
    global _ENGINE
    if _ENGINE is False:
        return None
    try:
        if _ENGINE is None:
            import ruaccent  # noqa: F401
            import ruphon  # noqa: F401
            from data import russian_g2p as _rg

            _ENGINE = _rg
        return _ENGINE.phonemize_russian(w).strip(" .")
    except Exception:
        _ENGINE = False
        return None


def _stress_word(w: str, nxt: str | None, prv: str | None = None,
                 yo_label: str | None = None, marked: str | None = None) -> tuple[str, str]:
    """-> ('+'-accented word, source tier).

    ``yo_label`` is the ё-tagger's verdict for this word ('YO'/'NO_YO') when the
    model ran; ``marked`` is the sentence with this word wrapped in <w></w>, for
    the homograph classifier."""
    lex = _lexicon()
    if "ё" not in w and w in lex["yo_hom"] and w not in _YO_LITERARY_E:
        if yo_label is not None:
            if yo_label == "YO":
                w = lex["yo_hom"][w]
        elif w == "все" and (nxt or prv):
            w = _vse(nxt)
            if w == "все":
                return "вс+е", "context"
    elif "ё" not in w and w in lex["yo"]:
        w = lex["yo"][w]
    vs = [i for i, c in enumerate(w) if c in VOWELS]
    if not vs:
        return w, "novowel"
    if "ё" in w:
        i = w.rfind("ё")
        return w[:i] + "+" + w[i:], "yo"
    if len(vs) == 1:
        return w[: vs[0]] + "+" + w[vs[0]:], "mono"
    if w in lex["hom"]:
        cands = [c for c in lex["hom"][w] if c < len(vs)]
        if marked is not None and _ctx().om_ok and len(cands) > 1:
            hyps = [w[: vs[c]] + "+" + w[vs[c]:] for c in cands]
            chosen = _ctx().choose_stress(marked, hyps)
            return chosen, "homograph-model"
        k, tier = lex["hom"][w][0], "homograph-prior"
        for key in (f"p|{prv or '<s>'}|{w}", f"n|{w}|{nxt or '</s>'}"):
            if key in lex["ctx"]:
                k, tier = lex["ctx"][key], "homograph-prior-ctx"
                break
    elif w in lex["acc"]:
        k, tier = lex["acc"][w], "dict"
    elif w in lex["train"]:
        k, tier = lex["train"][w], "train"
    else:
        return w, "ood"
    k = min(k, len(vs) - 1)
    return w[: vs[k]] + "+" + w[vs[k]:], tier


# ---------------------------------------------------------------------------
# tokenization and the public entry point
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"[а-яё]+|[a-z]+(?:'[a-z]+)?|[.,!?:;…]+", re.IGNORECASE)


def _latin_to_ipa(words: list[str]) -> list[str]:
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        be = EspeakBackend("en-us", preserve_punctuation=False, with_stress=True,
                           language_switch="remove-flags")
        out = be.phonemize([" ".join(words)], separator=Separator(phone="", word=" ", syllable=""))[0]
        return out.split() if len(out.split()) == len(words) else [out.strip()]
    except Exception:
        return words


def phonemize_russian(text: str) -> str:
    """Russian text -> normalized IPA string in the training-label conventions."""
    from data.text_vocab import normalize_text

    _REPORT.clear()
    text = unicodedata.normalize("NFC", text)
    for sign, word in _SIGNS.items():
        text = text.replace(sign, f" {word} ")
    text = _expand_numbers(text)
    text = text.replace("…", ".").replace("́", "")
    toks = _TOKEN.findall(text)
    out: list[str] = []
    latin_run: list[str] = []

    def flush_latin():
        if latin_run:
            ipas = _latin_to_ipa(latin_run)
            for w, p in zip(latin_run, ipas):
                _REPORT.append((w, "latin", p))
            out.extend(ipas)
            latin_run.clear()

    words = [t.lower() for t in toks]
    # the sentence as the context models saw it in RUAccent: lower case, words
    # space-separated, punctuation glued to the preceding word
    offsets, parts, pos = [], [], 0
    for t in words:
        glue = t[0] in _KEEP_PUNCT + "…" and parts
        if parts and not glue:
            pos += 1
        offsets.append(pos)
        parts.append(("" if (glue or not parts) else " ") + t)
        pos += len(t)
    sentence = "".join(parts)
    lex = _lexicon()
    yo = {}
    n_words = sum(1 for t in words if t[0] not in _KEEP_PUNCT)
    # the tagger needs a sentence: on 1-2 words it tags чем/лет/перед as чём/лёт/перёд
    if _ctx().yo_ok and n_words >= 3 and any(t in lex["yo_hom"] for t in words):
        lab = _ctx().yo_labels(sentence)
        yo = {i: lab.get(offsets[i]) for i, t in enumerate(words) if t in lex["yo_hom"]}

    def marked(i: int) -> str:
        a = offsets[i]
        return sentence[:a] + "<w>" + words[i] + "</w>" + sentence[a + len(words[i]):]

    for i, tok in enumerate(words):
        if tok[0] in _KEEP_PUNCT:
            flush_latin()
            p = "".join(ch for ch in tok if ch in _KEEP_PUNCT)
            if out and p:
                # collapse runs like '?!' '...' onto the labels' single-mark style
                out[-1] = out[-1].rstrip(_KEEP_PUNCT) + (p if len(set(p)) > 1 and p[0] in "?!" else p[0])
            continue
        if re.match(r"[a-z]", tok):
            latin_run.append(tok)
            continue
        flush_latin()
        nxt = next((x for x in words[i + 1:] if x[0] not in _KEEP_PUNCT), None)
        prv = next((x for x in reversed(words[:i]) if x[0] not in _KEEP_PUNCT), None)
        acc, tier = _stress_word(tok, nxt, prv, yo.get(i),
                                 marked(i) if tok in lex["hom"] or (tok in lex["yo_hom"]) else None)
        if tier == "ood":
            eng = _engine_word(tok)
            if eng:
                _REPORT.append((tok, "ood-engine", eng))
                out.append(eng)
                continue
            k = _suffix_stress(tok, _lexicon())
            vs = [j for j, c in enumerate(tok) if c in VOWELS]
            acc = tok[: vs[k]] + "+" + tok[vs[k]:] if vs else tok
            tier = "ood-suffix"
        ipa = word_to_ipa(acc)
        _REPORT.append((tok, tier, ipa))
        out.append(ipa)
    flush_latin()
    s = " ".join(x for x in out if x)
    return normalize_text(s, apply_hebrew_fixes=False) if s else ""


def last_report() -> list[tuple[str, str, str]]:
    """(word, tier, ipa) for every word of the last ``phonemize_russian`` call."""
    return list(_REPORT)
