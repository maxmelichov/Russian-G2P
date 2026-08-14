"""Pure-string tests for the Russian G2P front end.

Deliberately model-free: everything here runs without ruaccent/ruphon (and
without their ~1.5 GB of ONNX weights), so it can run in the training venv and
in CI. The model-backed stages are covered by spot-checking real sentences with
``scripts/phonemize_russian.py --text``.

    python -m pytest tests/ -q      # or: python tests/test_russian_g2p.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.russian_g2p import (
    apply_word_overrides,
    mark_yo_stress,
    remap_ruphon_ipa,
    to_vocab_ipa,
)
from data.text_vocab import PAD_ID, text_to_indices, unknown_symbols


# ---------------------------------------------------------------- mark_yo_stress


def test_yo_single():
    assert mark_yo_stress("всё") == "вс+ё"
    assert mark_yo_stress("её ещё") == "е+ё ещ+ё"


def test_yo_respects_existing_accent():
    """RUAccent already resolved this word -- do not add a second '+'."""
    assert mark_yo_stress("сам+олёт") == "сам+олёт"
    assert mark_yo_stress("вс+ё") == "вс+ё"


def test_yo_compound_marks_the_last():
    """Primary stress in a multi-ё compound is on the LAST ё, not the first."""
    assert mark_yo_stress("трёхколёсный") == "трёхкол+ёсный"
    assert mark_yo_stress("четырёхзвёздочный") == "четырёхзв+ёздочный"
    assert mark_yo_stress("трёхвёдерный") == "трёхв+ёдерный"
    # Exactly one primary stress, always.
    assert mark_yo_stress("трёхколёсный").count("+") == 1


def test_yo_capital():
    assert mark_yo_stress("Ёлка") == "+Ёлка"
    assert mark_yo_stress("Всё готово") == "Вс+ё готово"


def test_yo_non_space_whitespace():
    """split(' ') merged newline-separated words into one token."""
    assert mark_yo_stress("всё\nещё") == "вс+ё\nещ+ё"
    assert mark_yo_stress("всё\tещё") == "вс+ё\tещ+ё"


def test_yo_preserves_whitespace_runs():
    assert mark_yo_stress("а  б") == "а  б"
    assert mark_yo_stress(" всё ") == " вс+ё "


def test_yo_noop_without_yo():
    assert mark_yo_stress("привет мир") == "привет мир"


# -------------------------------------------------------------- remap_ruphon_ipa


def test_remap_tilde_affricates():
    assert remap_ruphon_ipa("t~sena") == "ʦena"
    assert remap_ruphon_ipa("t~ɕas") == "ʧas"
    assert remap_ruphon_ipa("d~ʐem") == "ʤem"


def test_remap_stress_mark():
    assert remap_ruphon_ipa("vɐ'da") == "vɐˈda"
    assert "'" not in remap_ruphon_ipa("z'amək")


def test_remap_real_tie_bars():
    """U+0361/U+035C forms must fold too, including the Russian-only ones that
    text_vocab's affricate pass does not know."""
    assert remap_ruphon_ipa("t͡sena") == "ʦena"
    assert remap_ruphon_ipa("t͡ɕas") == "ʧas"       # t͡ɕ -- not in text_vocab
    assert remap_ruphon_ipa("mʊt͡ʂinə") == "mʊʧinə"  # t͡ʂ -- not in text_vocab
    assert remap_ruphon_ipa("d͡ʑem") == "ʤem"
    assert remap_ruphon_ipa("t͜sena") == "ʦena"


def test_remap_residual_tie_dropped():
    """An unnamed tie-bar keeps both segments and loses only the bar."""
    assert remap_ruphon_ipa("p~f") == "pf"
    assert remap_ruphon_ipa("p͡f") == "pf"
    assert "͡" not in remap_ruphon_ipa("p͡f")


def test_vocab_ipa_has_no_oov():
    """The whole point of the remap: nothing reaches the model as PAD."""
    for ipa in ("t͡ɕas", "t~ɕas", "mʊt͡ʂinə", "vɐ'da", "fsʲ'ɵ"):
        assert unknown_symbols(to_vocab_ipa(ipa)) == [], ipa


# ------------------------------------------------------------ apply_word_overrides


def test_override_applies_on_source():
    assert apply_word_overrides("всё", "fsʲe") == "fsʲˈɵ"


def test_override_keeps_punctuation():
    assert apply_word_overrides("всё.", "fsʲe.") == "fsʲˈɵ."
    assert apply_word_overrides("Всё,", "fsʲe,") == "fsʲˈɵ,"


def test_override_uses_accented_form():
    """The corpus usually spells всё as все; only RUAccent's output has the ё.

    Keying on the raw source missed exactly the rows the override exists for.
    """
    assert apply_word_overrides("все", "fsʲe", "вс+ё") == "fsʲˈɵ"
    assert apply_word_overrides("и все.", "i fsʲe.", "и вс+ё.") == "i fsʲˈɵ."


def test_override_does_not_fire_on_genuine_vse():
    """все ('all') keeps /fsʲe/ -- that is its correct reading."""
    assert apply_word_overrides("все", "fsʲe", "вс+е") == "fsʲe"


def test_override_bails_on_misalignment():
    """Unequal token counts mean the alignment is untrustworthy: return as-is."""
    assert apply_word_overrides("всё и всё", "fsʲe") == "fsʲe"
    # A bad `accented` must not poison a source-based match either.
    assert apply_word_overrides("всё", "fsʲe", "вс+ё и вс+ё") == "fsʲˈɵ"


def test_override_noop_when_absent():
    assert apply_word_overrides("привет", "prʲɪvʲet") == "prʲɪvʲet"


# ------------------------------------------------------------------- text_vocab


def test_oov_detection():
    assert unknown_symbols("vɐˈda") == []
    assert unknown_symbols("привет") == ["п", "р", "и", "в", "е", "т"]


def test_oov_raises_when_asked():
    try:
        text_to_indices("привет", on_oov="raise")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for Cyrillic input")


def test_oov_pads_by_default():
    """Historical behaviour, kept for the training hot path."""
    assert text_to_indices("привет") == [PAD_ID] * 6


def test_cedilla_is_one_token_either_way():
    """NFD used to split precomposed ç into c + U+0327 -- two tokens for one
    phoneme, depending only on how upstream happened to encode it."""
    assert text_to_indices(to_vocab_ipa("c\u0327")) == text_to_indices(
        to_vocab_ipa("\u00e7")
    )


def test_stress_mark_is_its_own_token():
    ids = text_to_indices("vɐˈda")
    assert PAD_ID not in ids
    assert len(ids) == 5  # v ɐ ˈ d a -- the piper vocab is per-symbol by design


def test_palatalization_and_length_survive():
    for sym in ("ʲ", "ː", "ˌ"):
        assert text_to_indices(sym) != [PAD_ID]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
