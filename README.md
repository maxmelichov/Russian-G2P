# Russian G2P

Cyrillic → narrow IPA for TTS, on a fixed 256-token phoneme vocabulary.

Extracted from a multilingual 44.1 kHz TTS system, where Russian shares one
acoustic space (and one phoneme table) with Hebrew, Yiddish, English, German,
Italian and Spanish. Everything here is the Russian half of that pipeline.

```
Солнце уже село за старый замок.  →  sˈonʦə ʊʐˈɛ sʲˈeɫə za stˈarɨj zˈamək.
Он повесил замок на дверь.        →  on pɐvʲˈesʲɪɫ zɐmˈok nˈa dvʲerʲ.
```

Same word, two readings, resolved from context: за́мок (castle) vs замо́к (lock).

It runs on `onnxruntime` + `tokenizers`. There is no `transformers` dependency
and no neural phonemizer.

## How it works

Three stages, in `data/russian_phonemizer.py` and `data/russian_g2p_rules.py`.

1. **Stress and ё: a dictionary plus context models.** The lexicon comes from
   RUAccent's 3.19 M-form, Zaliznyak-derived dictionary. Two cases need more
   than a lookup. Words that exist both with and without ё (все/всё, сел/сёл)
   go to RUAccent's ё-tagger. Stress homographs (за́мок/замо́к, де́ла/дела́,
   го́ду/году́) go to RUAccent's `turbo3.1` homograph classifier. Both are ONNX
   graphs with a `tokenizer.json`, run here directly with onnxruntime and
   tokenizers. Before this stage, the text is normalized: numbers, years,
   dates, decimals, `%` and currency become words, and dashes, quotes and
   brackets are dropped.
2. **Word → IPA: Moscow-norm rules.** These cover akanye/ikanye reduction
   (`ɐ ə ɪ ɨ`), palatalization including assimilative `sʲtʲ zʲdʲ zʲnʲ`,
   final devoicing and voicing assimilation, ё → `ɵ`, `ʉ` and `æ` between
   soft consonants, `-ого/-его → və`, `-ться/-тся → ʦə` (`ʦːə` after the
   stress), silent consonants (солнце, праздник, чувство), orthoepic `ʂn/ʂt`
   (конечно, что), `сч/щ → ɕː`, and hard consonants in loanwords (кафе, тест,
   компьютер). The target is Wiktionary's ru-pron standard, the transcription
   RUPhon was trained to imitate. Each word is phonemized on its own; no
   sandhi across word boundaries.
3. **Vocab remap.** The output is written directly in the vocabulary's symbols:
   `ʦ ʧ ʤ` ligatures, `ˈ` right before the stressed vowel (before the `j` of
   an iotated vowel), `ɫ`, `Cː`. `data/text_vocab.normalize_text` is the last
   step. Nothing comes out that maps to PAD; the tests enforce this.

The rest of the text handling:

- **Punctuation.** `. , ! ? : ;` are kept on the preceding word.
- **Latin words.** Latin-script words are read with eSpeak `en-us` through
  `phonemizer`, if it is installed.
- **Out-of-dictionary words.** These are mostly names. They get a stress guess
  from the dictionary's majority stress for the word ending, or from a corpus
  lexicon if you built one with `--labels`.
- **Which tier was used.** `last_report()` (or `--report`) shows which tier
  placed the stress on every word.

```
$ python scripts/phonemize_russian.py --report --text "Он повесил замок на дверь."
  ipa: on pɐvʲˈesʲɪɫ zɐmˈok nˈa dvʲerʲ.
    он         mono              on
    повесил    dict              pɐvʲˈesʲɪɫ
    замок      homograph-model   zɐmˈok
    ...
```

## Install and build the lexicon

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # + phonemizer for Latin words
.venv/bin/python scripts/build_ru_lexicon.py                   # ~45 s, writes ru_lexicon/ (~375 MB, gitignored)
.venv/bin/python -m pytest -q tests/
```

`build_ru_lexicon.py` downloads RUAccent's dictionaries and its two context
models from the `ruaccent/accentuator` Hub repo. It then writes plain
`.tsv.gz` tables and copies the ONNX models. Options:

- `--src DIR` reuses an installed `ruaccent` package directory instead of
  downloading.
- `--labels meta.csv` adds corpus priors from a CSV with `text,ipa` columns:
  - a stress lexicon for names that are not in the dictionary;
  - a fallback homograph order, used when the context models are missing.

Set `RU_LEXICON_DIR` to keep the lexicon somewhere else.

## Use

```bash
python scripts/phonemize_russian.py --text "В 1945 году закончилась война, и 9 мая стал праздником."
#  ipa: v tˈɨsʲɪʧə dʲɪvʲɪʦˈot sˈorək pʲˈatəm ɡɐdˈu zɐkˈonʲʧɪɫəsʲ vɐjnˈa, i dʲɪvʲˈatəjə mˈajə staɫ prˈazʲnʲɪkəm.
python scripts/phonemize_russian.py --csv corpus.csv --column text --workers 8   # adds an ipa column
```

```python
from data.russian_phonemizer import phonemize_russian
phonemize_russian("Он повесил замок на дверь.")   # 'on pɐvʲˈesʲɪɫ zɐmˈok nˈa dvʲerʲ.'
```

About 13 sentences/s per process. Each process holds the lexicon (~1.5 GB RAM).

## Accuracy

The gold set is `tests/gold.tsv`: 604 words and 67 sentences (three of them
with digits). The reference IPA is en.wiktionary's standard ru-pron line; each
row records its URL and the raw IPA. It is converted to this notation:

- tie-bar affricates become ligatures;
- `ˈ` moves from the syllable onset to the vowel;
- optional segments `⁽ʲ⁾ (ː) (j)` expand to both variants;
- dialect lines are dropped.

In sentences, homograph stress is fixed by meaning.

| system | words exact | stress (polysyllables) | sentences exact | sentence PER |
| --- | --- | --- | --- | --- |
| **this engine** | **604/604** | **464/464** | **67/67** | **0.00%** |
| RUAccent + RUPhon (the legacy engine) | 565/604 | 461/464 | 44/67 | 3.4% |
| eSpeak-ng `ru` | 24/604 | 299/464 | 0/67 | 45.8% |

The gold set was also the development set. As a held-out check, I scraped 96
random corpus words that were never used during development: this engine got
96/96, the legacy engine 95/96.

The legacy engine's errors are systematic:

- **ɵ in unmarked monosyllables:** нет → nʲɵt, без → bʲɵs, тем → tʲɵm.
- **Wrong homograph readings:** она → ˈonə (the river). It showed up in
  ~5,000 of 50,275 corpus sentences.
- **Orthoepic `ʂn` read as `ʧn`:** конечно → kɐnʲˈeʧnə.
- **Final devoicing without `ɣ → x`:** бог → bok.
- **Digits spelled letter by letter:** "1945" → `æf`.

eSpeak does not reduce vowels and writes ы as `y`.

## Known residuals

- **Homographs.** They are only as good as RUAccent's context classifier. It
  gets most right, but not all; долгу and свечи were missed in a 40-sentence
  corpus sample, and the legacy engine misses them too.
- **ё/е pairs.** The ё-tagger is accepted at probability ≥ 0.8. At argmax it
  turned "и сел у окна" into сёл. далеко/недалеко are always the literary е
  form.
- **Loanwords with a hard consonant before е.** These come from a ~40-stem
  list (`HARD_STEMS`, plus whole-word `EXCEPTIONS`). Unlisted loans get a
  palatalized consonant.
- **Unknown names.** Stress is guessed by suffix analogy, so an unseen name can
  be stressed wrongly. `--labels` fixes the ones your corpus contains.
- **Latin words.** They go through English eSpeak and are not adapted to
  Russian. Without `phonemizer` they stay as Latin letters, which are valid
  vocab tokens but are spelled out.
- **Numbers.** Only years (before год/года/году…) and dates (before a month)
  become ordinals. Other numbers are nominative cardinals and are not inflected
  for case ("с 5 людьми" → пять).
- **No cross-word sandhi.** "в часы" stays `v ʧɪsˈɨ`. This is deliberate: it
  is the convention of the TTS labels this was built for.

## Layout

| file | |
| --- | --- |
| `data/russian_phonemizer.py` | text normalization, stress/ё (dictionary + ONNX context models), public `phonemize_russian` |
| `data/russian_g2p_rules.py` | one stressed word → IPA, Moscow-norm rules and exception lists |
| `data/text_vocab.py` | the 256-token IPA vocabulary the output has to land in (kept identical to the parent TTS repo) |
| `scripts/build_ru_lexicon.py` | builds `ru_lexicon/` from RUAccent's Hub repo; standalone |
| `scripts/phonemize_russian.py` | `--text` / `--csv` driver for the engine |
| `tests/gold.tsv`, `tests/test_gold.py` | the gold set and its test |
| `data/russian_g2p.py`, `scripts/phonemize_russian_legacy.py`, `tests/test_legacy_remap.py` | the legacy RUAccent + RUPhon engine (needs `requirements-legacy.txt`, transformers<5); kept because downstream code loads `data/russian_g2p.py` by path, and used as an optional out-of-dictionary fallback when importable |

## Vocabulary

`data/text_vocab.py` is the single source of truth for text → ids: 256 tokens
built on the Piper phoneme set, `PAD=0`, `BOS=1`, `EOS=2`. Unknown characters
map silently to `PAD`; they do **not** raise. Validate coverage when adding
material:

```python
from data.text_vocab import CHAR_TO_ID, normalize_text
ipa = normalize_text(ipa, apply_hebrew_fixes=False)
assert not {c for c in ipa if c not in CHAR_TO_ID}
```

## Credits

| project | author | license | role |
| --- | --- | --- | --- |
| [RUAccent](https://github.com/Den4ikAI/ruaccent) | Den4ikAI | MIT | The stress dictionary, the ё dictionaries, and the two ONNX context models (homograph classifier `turbo3.1`, ё-homograph tagger), from [`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator). |
| [Wiktionary](https://en.wiktionary.org) ru-pron | Wiktionary contributors | CC BY-SA | The pronunciation standard the rules implement and the gold set is drawn from. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime), [tokenizers](https://github.com/huggingface/tokenizers) | Microsoft, Hugging Face | MIT, Apache-2.0 | Run the context models. |
| [Piper](https://github.com/rhasspy/piper) | rhasspy | MIT | `data/text_vocab.py` is built on Piper's phoneme set. |
| [RUPhon](https://github.com/Den4ikAI/ruphon) | Denis Petrov, Ivan Shivalov | Apache-2.0 | Legacy engine only. |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) via [phonemizer](https://github.com/bootphon/phonemizer) | | GPL-3.0 | Optional, for Latin-script words only. |
