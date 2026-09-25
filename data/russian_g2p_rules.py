"""Deterministic Russian G2P: one stress-marked Cyrillic word -> IPA.

The target is Wiktionary's ru-pron standard (the transcription RUPhon was
trained to imitate), written in the notation of the TTS labels this repo
produces. Unlike RUPhon, which is a per-word neural model, the rules do not
make its systematic errors: е in an unmarked monosyllable read as ɵ
(нет -> nʲɵt, без -> bʲɵs, тем -> tʲɵm), конечно -> kɐnʲˈeʧnə, бог -> bok,
третий -> trˈɛtʲɪj. ``tests/gold.tsv`` is the reference set.

CONVENTIONS (measured on the labels, identical to Wiktionary's):

* A word is phonemized IN ISOLATION: no voicing or palatalization across word
  boundaries ("в часы" -> ``v ʧɪsˈɨ``, "из денег" -> ``is dʲˈenʲɪk``). Every label
  was made that way, so the model has learned the sandhi itself. Vowelless
  prepositions (в к с б ж) are left undevoiced for the same reason.
* ``ˈ`` goes immediately before the stressed vowel; when the stressed letter is an
  iotated е/ё/ю/я surfacing as j+V it goes before the ``j`` (моей ``mɐˈjej``,
  даёт ``dɐˈjɵt``, если ``ˈjesʲlʲɪ``).
* Polysyllables carry exactly one ``ˈ``. Monosyllables carry none, except the few
  that RUAccent marked in most label occurrences (``MONO_STRESSED``) and всё.
* Wiktionary's optional segments are realised: ⁽ʲ⁾ -> ʲ (впечатление ``fʲpʲ``),
  (ː) -> ː (именно ``nː``), (t) -> t (интеллигентский ``ntsk``).
* Affricates use the vocab ligatures (ʦ ʧ; дж -> ``ʤʐ`` as in Wiktionary's
  d͡ʐʐ); щ/сч -> ``ɕː``; hard л -> ``ɫ``; long consonants -> ``Cː``.

REDUCTION (Moscow norm): after a hard consonant unstressed а/о -> ``ɐ`` in the
syllable right before the stress and word-initially, ``ə`` elsewhere; after a
soft consonant unstressed а/е/я -> ``ɪ``, but я/а -> ``ə`` in the endings
-я -ям -ях -ями -ят -ятся; word-final е -> ``e`` (``ə`` in -ое and after ц).
After ж/ш/ц unstressed е -> ``ɨ``, and а -> ``ɨ`` before a soft consonant
(жалеть ``ʐɨlʲˈetʲ``, двадцать ``dvˈaʦːɨtʲ``). Stressed ё after a soft consonant
or j -> ``ɵ``; у/ю between soft consonants -> ``ʉ``; stressed а/я between soft
consonants -> ``æ``.
"""

from __future__ import annotations

import re

VOWELS = "аеёиоуыэюяЭƐ"  # Э: an е that keeps the consonant hard; Ɛ: an unreducible э (loanwords)
IOTATED = "еёюя"
SOFT_SIGNS = "еёиюяь"
CYRILLIC_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_BASE = {
    "б": "b", "в": "v", "г": "ɡ", "д": "d", "ж": "ʐ", "з": "z", "к": "k", "л": "l",
    "м": "m", "н": "n", "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f", "х": "x",
    "ц": "ʦ", "ч": "ʧ", "ш": "ʂ", "щ": "ɕː", "й": "j",
    # placeholders written by _rewrite():
    "Ч": "ʧː", "Ц": "ʦː", "Д": "ʤʐ", "Щ": "ɕː", "Ж": "ʐː", "Ш": "ʂː", "Ƭ": "ʦ", "Ǳ": "ʣ", "Ŝ": "s", "Ç": "ʦ", "Ś": "ʂ", "ɕ": "ɕ",
    # hard-consonant placeholders (loanword stems, see HARD_STEMS)
    "T": "t", "D": "d", "N": "n", "S": "s", "Z": "z", "R": "r", "L": "ɫ", "M": "m", "F": "f", "P": "p", "B": "b", "V": "v",
}
HARD_ONLY = set("жшцЦДЖШÇŚŜTDNSZRLMFPBV")
PALATAL = set("чщйЧЩɕ")
HUSH = set("жшцЦЖШДÇŚ")
VOICED_TO_VOICELESS = {"b": "p", "d": "t", "ɡ": "k", "z": "s", "ʐ": "ʂ", "v": "f", "ʤʐ": "ʧʂ", "ʐː": "ʂː", "ʣ": "ʦ"}
VOICELESS_TO_VOICED = {"p": "b", "t": "d", "k": "ɡ", "s": "z", "ʂ": "ʐ", "f": "v", "ʧ": "ʤ", "ʦ": "ʣ"}
OBSTRUENTS = set(VOICED_TO_VOICELESS) | set(VOICELESS_TO_VOICED) | {"x", "ɕː", "ʧː", "ʦː", "ʂː", "ʧʂ"}
VOICING_TRIGGERS = {"b", "d", "ɡ", "z", "ʐ", "ʤʐ", "ʐː", "ʣ"}  # в does not voice what precedes it
PHONE_VOWELS = "aeiouɨɪəɐʊæɵʉɛ"
NO_PAL_MARK = {"ʧ", "ɕ", "ɕː", "j", "ʂ", "ʐ", "ʧː", "ʦː", "ʤʐ", "ʐː", "ʂː", "ʧʂ", "ʦ"}

# Monosyllables that carry ˈ in the labels in >= 50% of occurrences (RUAccent's
# habit; kept so inference stays on the training token distribution).
MONO_STRESSED = {"а", "на", "но", "по", "о", "ли", "то", "при", "хоть", "ко"}

# Clitics whose reduced vowel is what the labels (and Wiktionary) write.
CLITICS = {"же": "ʐɨ", "во": "vɐ", "нибудь": "nʲɪbʊtʲ"}

# Whole-word exceptions (plain lowercase orthography -> IPA): what the rules
# cannot derive -- orthoepic чн/чт -> ʂn/ʂt, г -> v/x, silent consonants, й-drop.
EXCEPTIONS: dict[str, str] = {
    "что": "ʂto", "чтобы": "ʂtˈobɨ", "чтоб": "ʂtop", "ничто": "nʲɪʂtˈo", "нечто": "nʲˈeʂtə",
    "конечно": "kɐnʲˈeʂnə", "скучно": "skˈuʂnə", "нарочно": "nɐrˈoʂnə",
    "какие": "kɐkʲˈijɪ",  # Wiktionary kɐˈkʲi(j)ɪ, unlike другие/такие/новые (-je)
    "скучный": "skˈuʂnɨj", "скучная": "skˈuʂnəjə", "скучное": "skˈuʂnəjə",
    "яичница": "jɪˈiʂnʲɪʦə", "яичницу": "jɪˈiʂnʲɪʦʊ", "яичницы": "jɪˈiʂnʲɪʦɨ",
    "скворечник": "skvɐrʲˈeʂnʲɪk",
    "разносчик": "rɐznˈoɕʧɪk",  # Wiktionary; but извозчик/заказчик/подписчик are ɕː
    "сегодня": "sʲɪvˈodʲnʲə", "сегодняшний": "sʲɪvˈodʲnʲɪʂnʲɪj",
    "сегодняшнего": "sʲɪvˈodʲnʲɪʂnʲɪvə", "сегодняшняя": "sʲɪvˈodʲnʲɪʂnʲɪjə",
    "сегодняшнее": "sʲɪvˈodʲnʲɪʂnʲɪjə", "сегодняшнем": "sʲɪvˈodʲnʲɪʂnʲɪm",
    "итого": "ɪtɐvˈo",
    "бог": "box", "господь": "ɡɐspˈotʲ", "господи": "ɡˈospədʲɪ",
    "пожалуйста": "pɐʐˈaɫstə",
    "лестница": "lʲˈesʲnʲɪʦə", "лестницы": "lʲˈesʲnʲɪʦɨ", "лестнице": "lʲˈesʲnʲɪʦɨ", "лестницу": "lʲˈesʲnʲɪʦʊ",
    "кафе": "kɐfˈɛ", "шоссе": "ʂɐsːˈɛ", "пюре": "pʲʊrˈɛ", "радио": "rˈadʲɪo", "шестьсот": "ʂɨsːˈot", "счёл": "ɕʧˈɵɫ", "кашне": "kɐʂnˈɛ", "резюме": "rʲɪzʲʉmˈɛ",
}

# Loanword stems whose consonant stays hard before е (Wiktionary marks these). A
# stem matches at the start of a word, so inflected forms follow. Capital Latin
# letters are hard-consonant placeholders (see _BASE).
HARD_STEMS = {
    "тест": "Tест", "отел": "оTел", "модел": "моDел", "компьютер": "компьюTер",
    "интернет": "инTерNет", "теннис": "Tенис", "темп": "Tемп", "проект": "проЭкт",
    "энерги": "ƐNерги", "фонетик": "фоNетик", "генетик": "геNетик", "свитер": "свиTер", "бутерброд": "буTерброд",
    "детектив": "деTекTив", "шедевр": "шеDевр", "тенденци": "TенDенци", "фортепиано": "форTепиано",
    "стенд": "сTенд", "шрапнел": "шрапNел", "модерн": "моDерн", "бизнес": "бизNес",
    "денди": "Dенди", "тенор": "Tенор", "эстет": "эсTет", "синтез": "синTез", "ателье": "аTелье",
    "протез": "проTез",
}

# -ого/-его words where г is a real г (not the adjectival ending).
_OGO_KEEP_G = {
    "много", "немного", "премного", "дорого", "недорого", "строго", "нестрого", "убого",
    "полого", "отлого", "ого", "разлого", "благо", "пирого", "итого",
}
# Numerals whose final -ое keeps a full [je] (the adjectival -ое is [jə]).
_OE_FULL = {"кое", "трое", "двое", "пятеро"}

# Prefixes after which a doubled consonant is long (от-т, бес-с, раз-з, с-с ...).
_GEM_PREFIXES = ("от", "под", "над", "пред", "без", "бес", "раз", "рас", "из", "ис", "вз", "вс",
                 "воз", "вос", "об", "в", "с")

# Orthographic cluster simplifications (Wiktionary: silent consonants).
_CLUSTERS = [
    ("стн", "сн"),     # честный, известный
    ("здн", "зн"),     # праздник, поздно, бездна
    ("стл", "Ŝл"),     # счастливый ɕːɪslʲˈivɨj (Ŝ: this s stays hard)
    ("лнц", "нц"),     # солнце
    ("рдц", "рц"),     # сердце
    ("рдч", "рч"),     # сердчишко
]


def _is_vowel(ch: str) -> bool:
    return ch in VOWELS


def split_stress(word: str) -> tuple[str, int | None]:
    """'зам+ок' -> ('замок', 3): index of the stressed vowel letter (or None)."""
    idx = None
    out = []
    for ch in word:
        if ch == "+":
            idx = len(out)
            continue
        out.append(ch)
    w = "".join(out)
    if idx is not None:
        j = idx
        while j < len(w) and not _is_vowel(w[j]):
            j += 1
        idx = j if j < len(w) else None
    return w, idx


def _vowel_positions(w: str) -> list[int]:
    return [i for i, c in enumerate(w) if _is_vowel(c)]


def stress_vowel_number(ipa: str) -> int | None:
    """Index (among vowels) of the vowel carrying ˈ in an IPA word, or None."""
    v, marked = 0, False
    for ch in ipa:
        if ch == "ˈ":
            marked = True
            continue
        if ch in PHONE_VOWELS:
            if marked:
                return v
            v += 1
    return None


def word_to_ipa(word: str, mark_mono: bool | None = None) -> str:
    """One Russian word -> IPA.

    ``word`` is Cyrillic with '+' before the stressed vowel (RUAccent convention);
    ё implies its own stress. Other characters are dropped. ``mark_mono``
    forces (True) or suppresses (False) ˈ on a monosyllable; None applies the
    label convention.
    """
    word = word.lower().replace("́", "")
    w, stress = split_stress(word)
    keep = [i for i, c in enumerate(w) if c in CYRILLIC_LETTERS]
    if stress is not None:
        stress = keep.index(stress) if stress in keep else None
    w = "".join(w[i] for i in keep)
    if not w:
        return ""
    vpos = _vowel_positions(w)
    if stress is None or stress not in vpos:
        stress = w.rfind("ё") if "ё" in w else (vpos[0] if len(vpos) == 1 else None)
    nvow = len(vpos)
    # ё monosyllables are unmarked in the labels (пьёт pʲjɵt, днём dʲnʲɵm: 950 of 952
    # occurrences), except всё, which the old word override always wrote fsʲˈɵ
    mono_mark = mark_mono if mark_mono is not None else (w in MONO_STRESSED or w == "всё")

    if nvow == 0:
        # vowelless prepositions / particles: orthographic value, no devoicing
        # (в v, к k, с s, б b as in the labels), except the enclitic ж, which
        # the labels write devoiced (282/282: вот ж -> ʂ)
        return "ʂ" if w == "ж" else "".join(_BASE[c] for c in w if c in _BASE)
    if w in CLITICS and not (nvow == 1 and mono_mark):
        return CLITICS[w]
    if w in EXCEPTIONS:
        exc = EXCEPTIONS[w]
        k = vpos.index(stress) if stress in vpos else None
        if nvow <= 1 or k is None or stress_vowel_number(exc) == k:
            return exc if (nvow > 1 or mono_mark) else exc.replace("ˈ", "")

    s, stress, prefix_len = _rewrite(w, stress)
    segs = _segments(s, stress)
    _voicing(segs)
    ipa = _render(segs, prefix_len)
    if nvow <= 1 and not mono_mark:
        ipa = ipa.replace("ˈ", "")
    return ipa


def _sub_tracking(s: str, stress: int | None, src: str, dst: str, anchored: bool = False):
    """Replace src->dst everywhere (or only at the start), keeping ``stress`` on the same vowel."""
    pos = s.find(src)
    while pos >= 0:
        if anchored and pos != 0:
            break
        s = s[:pos] + dst + s[pos + len(src):]
        if stress is not None:
            if stress >= pos + len(src):
                stress -= len(src) - len(dst)
            elif stress >= pos:
                nb = sum(1 for c in src[: stress - pos] if _is_vowel(c))
                vi = [i for i, c in enumerate(dst) if _is_vowel(c)]
                stress = pos + (vi[nb] if nb < len(vi) else stress - pos)
        pos = s.find(src, pos + len(dst))
    return s, stress


def _rewrite(w: str, stress: int | None):
    """Orthographic rewrites that the phone rules then treat as ordinary letters."""
    s = w
    for stem, rep in HARD_STEMS.items():
        if s.startswith(stem):
            s, stress = _sub_tracking(s, stress, stem, rep, anchored=True)
            break
    if "чувств" in s or s.startswith("здравств"):
        s, stress = _sub_tracking(s, stress, "вств", "ств")
    for src, dst in _CLUSTERS:
        s, stress = _sub_tracking(s, stress, src, dst)
    # -ого / -его (and participles in -ого/-егося): г -> в
    if len(s) > 3 and (s.endswith("ого") or s.endswith("его")) and w not in _OGO_KEEP_G:
        s = s[:-2] + "в" + s[-1]
    elif len(s) > 6 and (s.endswith("огося") or s.endswith("егося")):
        s = s[:-4] + "в" + s[-3:]
    elif s in ("его", "кого", "того", "чего", "сего"):
        s = s[:-2] + "в" + s[-1]
    # -ться / -тся: ʦː right after the stressed vowel, ʦ elsewhere
    for end in ("ться", "тся"):
        if s.endswith(end) and len(s) > len(end):
            after_stress = stress is not None and stress == len(s) - len(end) - 1
            # Ç marks the infinitive -ться, whose preceding я stays [ɪ] (надеяться),
            # unlike the 3pl -ятся [ə] (становятся)
            s = s[: -len(end)] + ("Ц" if after_stress else ("Ç" if end == "ться" else "ц")) + "я"
            break
    for src, dst in (("сч", "Щ"), ("зч", "Щ"), ("жч", "Щ"), ("тч", "Ч"), ("дч", "Ч"),
                     ("тц", "Ц"), ("дц", "Ц"), ("дж", "Д"), ("тш", "чш"), ("дш", "чш"), ("тщ", "чщ"), ("дщ", "чщ"),
                     ("сж", "Ж"), ("зж", "Ж"), ("жж", "Ж")):
        s, stress = _sub_tracking(s, stress, src, dst)
    # сш/зш -> ʂʂ from two different letters; _geminate makes it long after the
    # stress or word-initially (высший, сшить), short before it (сумасшедший)
    for src in ("сш", "зш"):
        s, stress = _sub_tracking(s, stress, src, "Śш")
    # тс/дс -> ʦ + с (отсутствие ɐʦsˈuʦstvʲɪje); before к the с merges (детский
    # ʦk) except after н (интеллигентский ntsk, Wiktionary's optional (t)).
    m = re.search(r"(?<!н)[тд]с(?=к)", s)
    while m:
        s, stress = _sub_tracking(s, stress, m.group(0), "Ƭ")
        m = re.search(r"(?<!н)[тд]с(?=к)", s)
    s, stress = _sub_tracking(s, stress, "тьс", "Ƭ")  # пятьсот, девятьсот: ʦ (Wiktionary dʲɪvʲɪʦˈot)
    if not s.startswith("отст"):
        for src in ("тст", "дст"):
            s, stress = _sub_tracking(s, stress, src, "Ƭт")  # средство, отсутствие, представлять: ʦt
    s = re.sub(r"(?<!н)[тд](?=с)|(?<=н)[тд](?=с[^к])", "Ƭ", s)
    s = re.sub(r"[тд](?=з)", "Ǳ", s)
    prefix_len = 0
    for p in sorted(_GEM_PREFIXES, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p) + 1 and not _is_vowel(w[len(p)]) and w[len(p)] not in "ьъ":
            prefix_len = len(p)
            break
    return s, stress, prefix_len


def _is_soft(s: str, j: int, memo: dict) -> bool:
    """Whether the consonant letter at j is palatalized (ʲ), incl. assimilation."""
    if j in memo:
        return memo[j]
    c = s[j]
    res = False
    if c in HARD_ONLY or c in PALATAL or c not in _BASE:
        res = False
    else:
        nxt = s[j + 1] if j + 1 < len(s) else ""
        if nxt and nxt in SOFT_SIGNS:
            res = True
        elif nxt in _BASE and nxt not in PALATAL | HARD_ONLY:
            b, nb = _BASE[c], _BASE[nxt]
            if _is_soft(s, j + 1, memo):
                if b in ("s", "z") and nb in ("t", "d", "s", "z", "l"):
                    res = True  # сделать zʲdʲ, здесь zʲdʲesʲ, если sʲlʲ (Wiktionary s⁽ʲ⁾lʲ, labels sʲlʲ)
                elif b == "z" and nb == "n":
                    res = True  # жизнь ʐɨzʲnʲ, праздник zʲnʲ -- but снег snʲ, песню snʲ
                elif b in ("t", "d", "n", "ʦ") and nb in ("t", "d", "s", "z", "n"):
                    res = True
                elif b in ("v", "f", "m", "p", "b") and nb in ("p", "b"):
                    res = True
                elif b in ("ɡ", "k", "x") and nb in ("k", "ɡ"):
                    res = True
        elif nxt in ("ч", "щ", "Ч", "Щ") and _BASE[c] == "n":
            res = True
    memo[j] = res
    return res


def _segments(s: str, stress: int | None) -> list[dict]:
    n = len(s)
    vpos = _vowel_positions(s)
    sk = vpos.index(stress) if stress in vpos else None
    memo: dict = {}
    segs: list[dict] = []
    for i, c in enumerate(s):
        prv = s[i - 1] if i > 0 else ""
        if c in _BASE:
            if c == "й":
                segs.append({"t": "C", "p": "j", "soft": False, "letter": c})
                continue
            soft = _is_soft(s, i, memo)
            p = _BASE[c]
            if p == "l" and not soft:
                p = "ɫ"
            segs.append({"t": "C", "p": p, "soft": soft, "letter": c})
            continue
        if not _is_vowel(c):
            continue  # ь ъ
        k = vpos.index(i)
        stressed = sk is not None and k == sk
        rel = None if sk is None else k - sk
        if i == 0:
            left = "initial"
        elif _is_vowel(prv) or prv in "ьъ":
            left = "vowel"
        elif prv in HUSH:
            left = "hush"
        elif prv in HARD_ONLY:
            left = "hard"
        elif prv in PALATAL and prv != "й":
            left = "soft"
        elif prv == "й":
            left = "j"
        else:
            left = "soft" if _is_soft(s, i - 1, memo) else "hard"
        jglide = False
        cq = c
        if (c in IOTATED and left in ("initial", "vowel")) or (c in "ио" and prv == "ь"):
            segs.append({"t": "J"})
            jglide = True
            if c == "о":
                cq = "ё"  # ьо = /jo/ -> jɵ (синьор, бульон)
        p = _vowel(cq, stressed, rel, "j" if jglide else left, left, s, i, len(vpos) - 1 - k)
        # й + stressed vowel: the stress mark precedes the j, as for iotated
        # vowels (район rɐˈjon, майор mɐˈjor)
        segs.append({"t": "V", "p": p, "stressed": stressed, "jglide": jglide or prv == "й", "letter": c})
    return segs


def _right_soft(s: str, i: int) -> bool:
    j = i + 1
    if j >= len(s):
        return False
    c = s[j]
    if _is_vowel(c):
        return c in IOTATED
    if c in PALATAL:
        return True
    if c in _BASE:
        return _is_soft(s, j, {})
    return False


_SOFT_ENDINGS = ("", "м", "х", "ми", "т", "ця", "Ця")


def _vowel(c, stressed, rel, left, left_raw, s, i, vowels_after) -> str:
    if c == "Ɛ":
        return "ɛ"  # энергия ɛnˈɛrɡʲɪjə
    soft_left = left in ("soft", "j")
    nsoft = _right_soft(s, i)
    final = i == len(s) - 1
    if c == "Э":
        c = "э"
    if stressed:
        if c in "ая":
            return "æ" if (soft_left and nsoft) else "a"
        if c in "оё":
            if left in ("hush", "hard"):
                return "o"
            return "ɵ" if (c == "ё" or left == "soft") else "o"
        if c in "ую":
            return "ʉ" if (soft_left and nsoft) else "u"
        if c == "е":
            return "ɛ" if left in ("hush", "hard") else "e"
        if c == "э":
            return "ɛ"
        if c == "и":
            return "ɨ" if left == "hush" else "i"
        return "ɨ"  # ы
    pre = rel is not None and rel < 0
    pretonic1 = rel == -1
    if c == "ы":
        return "ɨ"
    if c == "и":
        return "ɨ" if left == "hush" else "ɪ"
    if c in "ую":
        return "ʉ" if (soft_left and nsoft) else "ʊ"
    if c == "э":
        return "ɪ" if left_raw in ("initial", "vowel") else ("ɨ" if pre else "ə")
    if c in "ао":
        if left == "soft":  # after ч щ
            return "ɪ" if (pre or s[i + 1:] not in _SOFT_ENDINGS) else "ə"
        if left == "hush" and c == "а" and nsoft and (
                (pretonic1 and ("жал" in s or "лошад" in s))  # жалеть ʐɨ, but шаги ʂɐ, жакет ʐɐ
                or (s[i - 1] in "цЦ" and s[i + 1:].startswith("ть"))):  # двадцать ʦːɨtʲ
            return "ɨ"
        if rel is None:
            return "ɐ" if left_raw == "initial" else "ə"
        if pre and (pretonic1 or left_raw == "initial"):
            return "ɐ"
        if pre and (s[i + 1:i + 2] in ("а", "о") or (i > 0 and s[i - 1] in "ао")):
            return "ɐ"
        return "ə"
    if c in "яеё":
        if left == "hush":
            if c == "е" and final and s[i - 1] in "цЦ" and not s.endswith("ице"):
                return "ə"
            return "ɨ" if c in "её" else ("ɐ" if pre else "ə")
        if left == "hard":
            return "ɨ" if c == "е" or pre else "ə"
        if pre:
            return "ɪ"
        if c == "е" and final:
            if i > 0 and s[i - 1] == "о" and s not in _OE_FULL:
                return "ə"  # adjectival -ое (такое tɐkˈojə)
            return "e"      # -ие/-ые/-ье/-ее (новые nˈovɨje, впечатление ...nʲɪje, более)
        if c == "я" and s[i + 1:] in _SOFT_ENDINGS and not s.endswith("десят"):
            return "ə"
        return "ɪ"
    return "ə"


def _voicing(segs: list[dict]) -> None:
    """Right-to-left: word-final devoicing and regressive voicing assimilation."""
    following = None
    for s in reversed(segs):
        if s["t"] == "C" and s["p"] in OBSTRUENTS:
            if following is None:
                s["p"] = VOICED_TO_VOICELESS.get(s["p"], s["p"])
            elif following["t"] == "C" and following["p"] in OBSTRUENTS:
                fp = following["p"]
                if fp in VOICING_TRIGGERS:
                    if s["p"] not in ("ʦ", "ʧ", "x"):
                        s["p"] = VOICELESS_TO_VOICED.get(s["p"], s["p"])
                elif fp != "v":
                    s["p"] = VOICED_TO_VOICELESS.get(s["p"], s["p"])
                    if s["letter"] == "г" and fp in ("k", "ʧ", "ʧː"):
                        s["p"] = "x"
        following = s


def _render(segs: list[dict], prefix_len: int) -> str:
    toks: list[str] = []
    for s in segs:
        if s["t"] == "J":
            toks.append("j")
        elif s["t"] == "V":
            if s["stressed"]:
                if s.get("jglide") and toks and toks[-1] == "j":
                    toks.insert(len(toks) - 1, "ˈ")
                else:
                    toks.append("ˈ")
            toks.append(s["p"])
        else:
            p = s["p"]
            if s["soft"] and (p not in NO_PAL_MARK or s["letter"] == "Ƭ"):
                p += "ʲ"
            toks.append(p)
    return _geminate(toks, segs, prefix_len)


def _cons_key(tok: str) -> str:
    if not tok or tok == "ˈ" or tok[0] in PHONE_VOWELS:
        return ""
    return tok.replace("ʲ", "").replace("ː", "").replace("ɫ", "l")


def _geminate(toks: list[str], segs: list[dict], prefix_len: int) -> str:
    """Collapse two identical consonants into one, long or short.

    Long: across a prefix boundary (от-т, бес-с), when the two came from
    different letters (отдать -> dː), and нн/мм after the stress (именно nː);
    short: other roots (класса, Россия), before a consonant, word-finally."""
    letters = [s for s in segs if s["t"] == "C"]
    res: list[str] = []
    ci = -1  # index into letters of the current consonant token
    stressed_seen = False
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "ˈ":
            stressed_seen = True
            res.append(t)
            i += 1
            continue
        is_cons = bool(_cons_key(t)) and t != "j"
        if is_cons:
            ci += 1
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if is_cons and nxt and nxt != "ˈ" and _cons_key(nxt) == _cons_key(t) and nxt != "j":
            a, b = letters[ci], letters[ci + 1]
            after = toks[i + 2] if i + 2 < len(toks) else ""
            before_cons = bool(after) and after != "ˈ" and after[0] not in PHONE_VOWELS
            a_idx = next(k for k, sg in enumerate(segs) if sg is a)
            at_prefix = bool(prefix_len) and sum(1 for sg in segs[: a_idx + 1] if sg["t"] != "J") == prefix_len
            if not after:
                long_ = False
            elif at_prefix:
                long_ = True
            elif before_cons:
                long_ = False
            elif a["letter"] != b["letter"]:
                long_ = stressed_seen or ci == 0 or a["letter"] in "ЖЧЦЩ"
            elif a["letter"] == "н" or (a["letter"] == "м" and stressed_seen):
                long_ = True
            elif ci == 0 and not res:
                long_ = True  # word-initial сс-, жж-
            else:
                long_ = False
            res.append(nxt if (not long_ or "ː" in nxt) else nxt + "ː")
            ci += 1
            i += 2
            continue
        res.append(t)
        i += 1
    return "".join(res)
