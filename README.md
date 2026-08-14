# Russian G2P

Cyrillic → narrow IPA for TTS, on a fixed 256-token phoneme vocabulary.

Extracted from a multilingual 44.1 kHz TTS system, where Russian shares one
acoustic space (and one phoneme table) with Hebrew, Yiddish, English, German,
Italian and Spanish. Everything here is the Russian half of that pipeline.

```
Солнце уже село за старый замок.  →  sˈonʦɨ ʊʐˈɛ sʲˈeɫə za stˈarɨj zˈamək.
Он повесил замок на дверь.        →  on pɐvʲˈesʲɪɫ zɐmˈok nˈa dvʲerʲ.
```

Same word, two readings, resolved from context: за́мок (castle) vs замо́к (lock).

## How it works

Three stages.

1. **RUAccent** (`turbo3.1`, dictionary on) places lexical stress and restores
   ё that the orthography omits. Emits the `+`-before-stressed-vowel convention.
2. **RUPhon** (`big`) turns `+`-accented Cyrillic into narrow IPA, applying the
   stress-conditioned vowel reduction that makes Russian sound Russian —
   `зам+ок → zɐmˈok` vs `з+амок → zˈamək`.
3. **Remap** (`remap_ruphon_ipa`) folds RUPhon's tie-bar affricates and ASCII
   stress mark onto the symbols the vocabulary already uses.

Stage 3 is pure string work with no model dependencies, so it is safe to import
from a training or inference process. Stages 1–2 are not: `ruaccent` and
`ruphon` pin `transformers<5`, which is why they are run offline in a throwaway
environment and the result is consumed as a precomputed column.

### Why not espeak-ng

espeak's `ru` voice is context-invariant, so `на двери висит замок` and
`на горе стоит замок` phonemize identically. It also cannot restore omitted ё
(~34% of rows in the corpus this was built for need it, and ё is always
stressed), and it writes ы as `/y/`, which collides with the German ü already
in the shared vocabulary.

### Why not nsu-ai/russian_g2p

Different lineage, and not substitutable here:

| | nsu-ai/russian_g2p | this |
| --- | --- | --- |
| output | custom labels, `['D0','I','A','L','O0','K']` | narrow IPA |
| stress | `+` must already be in the input, or dictionary lookup | neural homograph model + dictionary |
| homographs | caller supplies morph tags (`['ноги','NOUN Case=Gen\|...']`) | resolved from sentence context |
| ё restoration | no | yes |

The output format alone rules it out: nothing downstream of an IPA phoneme
table accepts `D0`/`O0`.

## Install

The model stack pins an old `transformers`, so give it its own environment:

```bash
uv venv /tmp/ruvenv --python 3.11
uv pip install --python /tmp/ruvenv/bin/python -r requirements.txt
```

## Use

One sentence:

```bash
/tmp/ruvenv/bin/python scripts/phonemize_russian.py \
    --text "Солнце уже село за старый замок." --out ru.json
```

```
Солнце уже село за старый замок.
  accented: С+олнце уж+е с+ело за ст+арый з+амок.
  ipa     : sˈonʦɨ ʊʐˈɛ sʲˈeɫə za stˈarɨj zˈamək.
```

A whole corpus — reads `<root>/metadata.csv`, writes `<root>/metadata_ipa.csv`
with an added `ipa` column:

```bash
/tmp/ruvenv/bin/python scripts/phonemize_russian.py --root /path/to/corpus --workers 24
```

Neither model exposes a batch API — `phonemize()` takes one string — so
throughput comes from processes, not batching. Each worker loads its own copy
of both models (~1.5 GB), so `--workers` is bounded by RAM, not cores.

One non-obvious detail is documented at length in the source: onnxruntime sizes
its thread pool from `SessionOptions.intra_op_num_threads` and **ignores**
`OMP_NUM_THREADS` for the default CPU EP. RUAccent builds four sessions and
RUPhon one, none passing SessionOptions, so 24 workers × 5 sessions × 32 threads
produced a load average of ~580 on 32 cores and 5 rows/s — half the
single-process rate. `_pin_single_thread()` patches the constructor, which is
the only lever short of forking the packages. It must run before `ruaccent` and
`ruphon` are imported, since they capture `InferenceSession` at import time.

From Python, without the corpus driver:

```python
from data.russian_g2p import phonemize_russian
phonemize_russian("Он повесил замок на дверь.")   # 'on pɐvʲˈesʲɪɫ zɐmˈok nˈa dvʲerʲ.'
```

## Layout

| file | |
| --- | --- |
| `data/russian_g2p.py` | the three stages; `remap_ruphon_ipa` is the model-free part |
| `data/text_vocab.py` | the 256-token IPA vocabulary the output has to land in |
| `scripts/phonemize_russian.py` | corpus driver and `--text` one-shot mode |

## Two corrections applied on top of the models

Both are narrow and both are justified by measurement, not taste.

**`mark_yo_stress`** — RUAccent only inserts `+` where stress is *ambiguous*, so
it leaves ё alone when the orthography already writes it. RUPhon needs the mark
to realise ё as `/ɵ/` and renders the bare letter as `/e/`. ё is always stressed
in Russian, so marking it is unconditionally correct. Without this, 445 of the
738 rows that spell ё out (60.3%) produced no `/ɵ/` at all.

**`_WORD_IPA_OVERRIDES`** — всё renders `/fsʲe/` however it is written (всё,
вс+ё, Вс+ё), and no respelling helps (фсё → `fsʲe`, всьо → `fsʲjɵ`). It cannot
be patched at the IPA level either, because `/fsʲe/` is the *correct* reading of
все ("all") and все/всё is precisely the ё distinction — so the substitution has
to know which source word it came from. Scoped tight: всё* is 453 of the 472
ё-tokens that phonemize wrong (96%); the rest occur 1–3 times each.

## Vocabulary

`data/text_vocab.py` is the single source of truth for text → ids: 256 tokens
built on the Piper phoneme set, `PAD=0`, `BOS=1`, `EOS=2`. Unknown characters
map silently to `PAD` — they do **not** raise — so validate coverage when adding
material:

```python
from data.text_vocab import CHAR_TO_ID, normalize_text
ipa = normalize_text(ipa, apply_hebrew_fixes=False)
assert not {c for c in ipa if c not in CHAR_TO_ID}
```

The full Russian inventory produced by this pipeline is covered, schwa included
(`ə`=59, `ɐ`=50, `ɨ`=73, `ɫ`=75, `ʲ`=119, `ʦ`=155, `ɵ`=85).

`remap_ruphon_ipa` must run **before** `normalize_text`, whose affricate pass
does not recognise the tilde tie-bar form.

## Credits

This repository is thin. Almost all of the linguistic work is done by other
people's open source, and the two Den4ikAI libraries in particular are what
make context-sensitive Russian G2P possible at all here.

### Runtime dependencies

| project | author | license | role |
| --- | --- | --- | --- |
| [RUAccent](https://github.com/Den4ikAI/ruaccent) | Den4ikAI | MIT | **Stage 1.** Stress placement, homograph resolution and ё restoration. Models on the Hub at [`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator); this pipeline uses `turbo3.1` with the dictionary enabled. |
| [RUPhon](https://github.com/Den4ikAI/ruphon) | Denis Petrov, Ivan Shivalov (© 2024) | Apache-2.0 | **Stage 2.** `+`-accented Cyrillic → narrow IPA with stress-conditioned vowel reduction. This pipeline uses the `big` model. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Microsoft | MIT | Executes both models. RUAccent builds four sessions and RUPhon one; the threading note above is about this runtime, not about the libraries. |
| [transformers](https://github.com/huggingface/transformers) / [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Hugging Face | Apache-2.0 | Tokenizers and model download. The `transformers<5` pin these libraries carry is the reason this runs in its own environment. |

### Phoneme inventory

| project | author | license | role |
| --- | --- | --- | --- |
| [Piper](https://github.com/rhasspy/piper) | rhasspy (Michael Hansen) | MIT | `data/text_vocab.py` is built on Piper's phoneme set — ids 0–156 are the Piper core, 157–244 are extensions added for the other languages in the parent system. |

### Referenced, not used

| project | role here |
| --- | --- |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) (GPL-3.0) | The obvious alternative front end, and the one this pipeline deliberately does *not* use for Russian — see "Why not espeak-ng" above. |
| [nsu-ai/russian_g2p](https://github.com/nsu-ai/russian_g2p) | NSU's Russian G2P, compared against above. Different output format and a different approach to stress; not a dependency. |

### Data

The measurements quoted in this README — the 60.3% ё figure, the 96% всё
figure, the 26 замок tokens — come from a Russian read-audiobook corpus of
LibriVox recordings ("Russian LibriSpeech"), ~50k utterances. The numbers are
reported so the corrections in `russian_g2p.py` can be checked rather than
taken on trust; the audio itself is not redistributed here.
