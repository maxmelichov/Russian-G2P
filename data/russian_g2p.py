"""
Russian G2P: Cyrillic -> narrow IPA for TTS, in the 256-token Piper vocab.

ARCHITECTURE: Two-Stage (accentuation -> phonemization -> vocab remap)
  1. RUAccent  resolves lexical stress AND restores omitted ё, from sentence
     context. Emits the '+'-before-stressed-vowel convention.
  2. RUPhon    turns '+'-accented Cyrillic into narrow IPA, applying the
     stress-conditioned vowel reduction that makes Russian sound Russian
     (зам+ок -> zɐm'ok vs з+амок -> z'amək).
  3. remap     folds RUPhon's tie-bar affricates and ASCII stress mark onto the
     symbols this project's vocab already uses.

Stage 1+2 need the ruaccent/ruphon packages, which pin transformers<5 and are
therefore kept OUT of the training environment: ``scripts/phonemize_russian.py``
runs them once offline and writes an ``ipa`` column, exactly as the
yiddish24-wav corpus ships a precomputed one. Only ``remap_ruphon_ipa`` -- which
is pure string work -- is imported by the training/inference path.

WHY NOT ESPEAK: espeak-ng's ru voice cannot disambiguate homographs at all (it
is context-invariant, so на двери висит замок and на горе стоит замок phonemize
identically), cannot restore ё that the orthography omits (~34% of rows in
russian_librispeech need it, and ё is always stressed), and writes ы as /y/ --
which collides with the German ü already in this vocab.
"""

from __future__ import annotations

import re

# =====================================================================
# STAGE 3: VOCAB REMAP
#
# RUPhon marks affricates with a tilde tie-bar (t~s) and stress with an ASCII
# apostrophe. Both are technically in the 256-token table already, but as the
# WRONG tokens: '~' is this vocab's tilde and "'" is its apostrophe, so stress
# would silently train into a punctuation embedding without ever raising an OOV.
# =====================================================================

# Tie-bar affricates -> the single ligatures this vocab uses. ч is /tɕ/ and ʧ is
# /tʃ/, so the mapping is lossy by design: it shares an embedding with the
# Yiddish and English affricate rather than sitting alone.
_AFFRICATES: dict[str, str] = {
    "t~s": "ʦ",   # ʦ  ц       -- matches yiddish_g2p, which emits ʦ directly
    "t~ɕ": "ʧ",   # ʧ  ч
    "t~ʂ": "ʧ",   # ʧ  тш across a morpheme boundary (rare)
    "d~z": "ʣ",   # ʣ  дз (loanwords)
    "d~ʑ": "ʤ",   # ʤ  дж
    "d~ʐ": "ʤ",   # ʤ  дж (retroflex realisation)
}

# RUPhon's stress mark (U+0027) -> the IPA primary stress every other language
# in this vocab uses (U+02C8).
_RUPHON_STRESS = "'"
_IPA_STRESS = "ˈ"

# RUPhon writes tie-bars as ASCII '~', but the same model emits real IPA tie
# bars (U+0361 COMBINING DOUBLE INVERTED BREVE, U+035C below) depending on
# version and on the stress_symbol argument. Fold both onto '~' first so the
# table below sees one form. text_vocab's affricate pass only knows t͡s/t͡ʃ, so
# the Russian-only t͡ɕ / t͡ʂ / d͡ʑ / d͡ʐ would otherwise survive into the vocab
# as three separate tokens (t, U+0361, ɕ).
_TIE_CHARS = re.compile("[͜͡]")

# Any tie-bar this table does not name: keep both segments, drop the bar.
_RESIDUAL_TIE = re.compile(r"(\S)~(\S)")


def remap_ruphon_ipa(text: str) -> str:
    """Fold RUPhon output onto this project's phoneme inventory.

    Pure string transformation with no model dependencies, so it is safe to
    import from the training and inference paths. Must run BEFORE
    ``text_vocab.normalize_text``, whose affricate pass does not recognise the
    tilde tie-bar form.
    """
    text = _TIE_CHARS.sub("~", text)
    for src, dst in _AFFRICATES.items():
        text = text.replace(src, dst)
    text = _RESIDUAL_TIE.sub(r"\1\2", text)
    return text.replace(_RUPHON_STRESS, _IPA_STRESS)


_WHITESPACE = re.compile(r"(\s+)")


def mark_yo_stress(accented: str) -> str:
    """Ensure every ё-bearing word carries an explicit stress mark.

    RUAccent only inserts '+' where stress is ambiguous, so it leaves ё alone when
    the orthography already writes it -- but RUPhon needs the mark to realise ё as
    /ɵ/ and renders the bare letter as /e/. ё is *always* stressed in Russian, so
    marking it is unconditionally correct.

    Without this, 445 of the 738 rows in russian_librispeech that spell ё out
    (60.3%) produced no /ɵ/ at all. ё-restoration was unaffected -- RUAccent marks
    the stress on ё it inserts itself -- so this only touches already-written ё,
    which includes very common words (всё, ещё, её).

    Three cases the naive ``replace("ё", "+ё", 1)`` got wrong:

    * **Compounds with more than one ё.** трёхколёсный, четырёхзвёздочный and
      трёхвёдерный carry PRIMARY stress on the LAST ё (-лё-, -звё-, -вё-); the
      earlier one is secondary. Marking the first put primary stress on the wrong
      syllable. There is no secondary-stress mark in RUPhon's '+' convention, so
      the earlier ё is left bare -- one primary is right, two would be invalid.
    * **Capital Ё.** Ёлка / Ёжик at sentence start contain no lowercase 'ё', so
      the word was skipped entirely and RUPhon read it as /e/.
    * **Newlines and tabs.** ``split(" ")`` treats "всё\\nещё" as one token, so
      only the first ё of the pair was ever marked.
    """
    parts = _WHITESPACE.split(accented)
    for i, word in enumerate(parts):
        if "+" in word:
            continue  # RUAccent already resolved this word's stress
        idx = max(word.rfind("ё"), word.rfind("Ё"))
        if idx >= 0:
            parts[i] = word[:idx] + "+" + word[idx:]
    return "".join(parts)


# Words RUPhon mispronounces no matter how they are marked. всё renders /fsʲe/
# whether written всё, вс+ё or Вс+ё, and no respelling helps (фсё -> fsʲe,
# всьо -> fsʲjɵ). It cannot be patched at the IPA level either: /fsʲe/ is the
# CORRECT reading of все ("all"), and все/всё is precisely the ё distinction --
# so the substitution has to know which source word it came from.
#
# Scoped deliberately tight. In russian_librispeech, всё* is 453 of the 472
# ё-tokens that phonemize wrong (96%); the rest occur 1-3 times each.
_WORD_IPA_OVERRIDES: dict[str, str] = {
    "всё": "fsʲˈɵ",
}


_STRIP_PUNCT = ".,!?;:\"'()«»—…-–[]{}"


def _override_key(token: str) -> str:
    """Orthographic lookup key: strip punctuation, RUAccent's '+', case-fold."""
    return token.strip(_STRIP_PUNCT).replace("+", "").lower()


def apply_word_overrides(source: str, ipa: str, accented: str | None = None) -> str:
    """Replace IPA tokens for source words in ``_WORD_IPA_OVERRIDES``.

    Token-aligned and fail-safe: if the word counts disagree, RUPhon did not emit
    one token per input word and the alignment cannot be trusted, so the IPA is
    returned untouched rather than corrupted.

    ``accented`` is RUAccent's output for the same string, and is the form the
    lookup should key on when it is available. Russian orthography routinely
    writes ё as е -- that is the whole reason RUAccent's ё-restoration is in this
    pipeline -- so a corpus row spelling всё as "все" never matched the override
    table when only the raw source was consulted, which is precisely the case the
    override exists to fix. The raw source stays as the fallback for when
    RUAccent retokenized and the alignment cannot be trusted.
    """
    src_tokens = source.split()
    ipa_tokens = ipa.split()
    if len(src_tokens) != len(ipa_tokens):
        return ipa
    acc_tokens = accented.split() if accented else []
    if len(acc_tokens) != len(src_tokens):
        acc_tokens = []
    changed = False
    for i, raw in enumerate(src_tokens):
        key = _override_key(acc_tokens[i]) if acc_tokens else _override_key(raw)
        replacement = _WORD_IPA_OVERRIDES.get(key)
        if replacement is None and acc_tokens:
            replacement = _WORD_IPA_OVERRIDES.get(_override_key(raw))
        if replacement is None:
            continue
        # Carry over any trailing punctuation the phonemizer kept.
        tail = ""
        while ipa_tokens[i] and ipa_tokens[i][-1] in ".,!?;:":
            tail = ipa_tokens[i][-1] + tail
            ipa_tokens[i] = ipa_tokens[i][:-1]
        ipa_tokens[i] = replacement + tail
        changed = True
    return " ".join(ipa_tokens) if changed else ipa


def to_vocab_ipa(text: str) -> str:
    """RUPhon output -> normalized IPA ready for ``text_to_indices``."""
    from data.text_vocab import normalize_text

    return normalize_text(remap_ruphon_ipa(text), apply_hebrew_fixes=False)


# =====================================================================
# STAGES 1-2: the model-backed front end (offline / inference only)
# =====================================================================

_ACCENTOR = None
_PHONEMIZER = None


def _load(device: str = "CPU", workdir: str | None = None):
    """Lazily build the RUAccent + RUPhon pair. Requires ruaccent and ruphon."""
    global _ACCENTOR, _PHONEMIZER
    if _ACCENTOR is None or _PHONEMIZER is None:
        try:
            from ruaccent import RUAccent
            from ruphon import RUPhon
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Russian raw-text G2P needs ruaccent and ruphon:\n"
                "  pip install ruaccent ruphon 'transformers<5'\n"
                "These pin transformers<5, so prefer running "
                "scripts/phonemize_russian.py in a separate venv and consuming "
                "its precomputed 'ipa' column."
            ) from exc

        accentor = RUAccent()
        accentor.load(
            omograph_model_size="turbo3.1",
            use_dictionary=True,
            tiny_mode=False,
        )
        phonemizer = RUPhon()
        kwargs = {"device": device}
        if workdir:
            kwargs["workdir"] = workdir
        _ACCENTOR, _PHONEMIZER = accentor, phonemizer.load("big", **kwargs)
    return _ACCENTOR, _PHONEMIZER


def accent_russian(text: str, device: str = "CPU", workdir: str | None = None) -> str:
    """Cyrillic -> '+'-accented Cyrillic with ё restored."""
    accentor, _ = _load(device, workdir)
    return mark_yo_stress(accentor.process_all(str(text).strip()))


def phonemize_russian(
    text: str, device: str = "CPU", workdir: str | None = None
) -> str:
    """Raw Cyrillic -> IPA on this project's phoneme inventory (unnormalized)."""
    accentor, phonemizer = _load(device, workdir)
    source = str(text).strip()
    accented = mark_yo_stress(accentor.process_all(source))
    ipa = remap_ruphon_ipa(phonemizer.phonemize(accented)).strip()
    return apply_word_overrides(source, ipa, accented)
