"""
Offline Russian phonemization: russian_librispeech metadata.csv -> + an 'ipa' column.

Runs RUAccent (stress + ё restoration) then RUPhon (narrow IPA), then folds the
output onto this project's 256-token vocab via data/russian_g2p.remap_ruphon_ipa.

Kept as a standalone script because ruaccent/ruphon pin transformers<5, which
would drag huggingface-hub below 1.0 in the training venv for no benefit. Run it
in its own environment; training then reads the precomputed column, exactly as
the yiddish24-wav corpus already ships one.

    uv venv /tmp/ruvenv --python 3.11
    uv pip install --python /tmp/ruvenv/bin/python ruaccent ruphon 'transformers==4.44.2'
    /tmp/ruvenv/bin/python scripts/phonemize_russian.py --workers 24

For a single sentence (the only supported way to synthesize ad-hoc Russian --
inference_helper has no 'ru' eSpeak entry and assumes --text is already IPA):

    /tmp/ruvenv/bin/python scripts/phonemize_russian.py \
        --text "Солнце уже село за старый замок." --out ru.json
    .venv/bin/python scripts/run_ipa_inference.py --ipa_json ru.json ...

Neither model exposes a batch API -- phonemize() takes one string -- so
throughput comes from processes, not batching. Each worker loads its own copy of
both models (~1.5 GB), so --workers is bounded by RAM, not cores.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.russian_g2p import (  # noqa: E402  (pure string work)
    apply_word_overrides,
    mark_yo_stress,
    remap_ruphon_ipa,
    to_vocab_ipa,
)
from data.text_vocab import unknown_symbols  # noqa: E402

DEFAULT_ROOT = "/home/maxm/AE_training_data_all/datasets_4AE_extracted/russian_librispeech"

# RUPhon can emit the IPA stress mark itself, which saves a remap pass.
IPA_STRESS = "ˈ"

_ACCENTOR = None
_PHONEMIZER = None


def _pin_single_thread():
    """Force every ONNX Runtime session in this process to a single thread.

    Parallelism here comes from processes, so each worker must be single-threaded
    or they thrash. Environment variables are NOT sufficient: onnxruntime sizes
    its CPU thread pool from SessionOptions.intra_op_num_threads and ignores
    OMP_NUM_THREADS for the default CPU EP. RUAccent builds four sessions and
    RUPhon one, none of them passing SessionOptions, so 24 workers x 5 sessions x
    32 threads produced a load average of ~580 on 32 cores and throughput of
    5 rows/s -- half the single-process rate. Patching the constructor is the
    only lever short of forking the packages.

    Must run BEFORE ruaccent/ruphon are imported: their modules do
    `from onnxruntime import InferenceSession` at import time, which would
    capture the unpatched class.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"

    import onnxruntime as ort

    if getattr(ort, "_lightblue_single_threaded", False):
        return
    _Original = ort.InferenceSession

    class _SingleThreaded(_Original):
        def __init__(self, *args, **kwargs):
            opts = kwargs.get("sess_options")
            if opts is None and len(args) > 1 and isinstance(args[1], ort.SessionOptions):
                opts = args[1]
            if opts is None:
                opts = ort.SessionOptions()
                kwargs["sess_options"] = opts
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            # Without this, ORT may still run independent graph branches on its
            # inter-op pool; one thread per worker is the whole point here.
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            super().__init__(*args, **kwargs)

    ort.InferenceSession = _SingleThreaded
    ort._lightblue_single_threaded = True


def _init_worker(device: str, workdir: str, accent_model: str, phon_model: str):
    """Load one RUAccent + RUPhon pair per worker process."""
    global _ACCENTOR, _PHONEMIZER
    _pin_single_thread()  # must precede the imports below
    from ruaccent import RUAccent
    from ruphon import RUPhon

    accentor = RUAccent()
    accentor.load(
        omograph_model_size=accent_model,
        use_dictionary=True,
        tiny_mode=False,
        device=device,
    )
    _ACCENTOR = accentor
    _PHONEMIZER = RUPhon().load(phon_model, workdir=workdir, device=device)


def _accent_one(text: str) -> str:
    """Cyrillic -> '+'-accented Cyrillic with ё restored and every ё marked."""
    return mark_yo_stress(_ACCENTOR.process_all(str(text).strip()))


def _phonemize_accented(source: str, accented: str) -> str:
    """Accented Cyrillic -> vocab-ready IPA, keyed on both forms for overrides."""
    ipa = _PHONEMIZER.phonemize(accented, stress_symbol=IPA_STRESS)
    return apply_word_overrides(source, remap_ruphon_ipa(ipa).strip(), accented)


def _phonemize_one(text: str) -> str:
    """Cyrillic -> vocab-ready IPA. Returns '' on failure so the row can be dropped."""
    try:
        source = str(text).strip()
        return _phonemize_accented(source, _accent_one(source))
    except Exception:
        return ""


def _phonemize_texts(args) -> int:
    """Ad-hoc mode: a few sentences straight to IPA, no corpus, no Pool.

    Exists because there was no way to synthesize an arbitrary Russian sentence.
    run_pt_inference/inference_helper have no 'ru' entry in their eSpeak map and
    treat --text as IPA already, so Cyrillic typed there is silently mapped to
    PAD by text_vocab rather than rejected. Going through RUAccent is what
    resolves homographs from context (за старый з+амок vs пов+есил зам+ок) --
    that disambiguation is the whole reason espeak is not used for Russian.
    """
    _init_worker(
        args.device,
        args.workdir or os.path.join(args.root, ".ruphon_models"),
        args.accent_model,
        args.phon_model,
    )
    results = []
    for text in args.text:
        # One accentuation pass, reused for the printout and the phonemization:
        # running RUAccent twice doubled the latency of the interactive path and
        # let the line labelled 'accented' drift from the one actually
        # phonemized whenever the model was non-deterministic.
        source = str(text).strip()
        try:
            accented = _accent_one(source)
            ipa = _phonemize_accented(source, accented)
        except Exception as exc:
            print(f"[Russian G2P] FAILED: {text}: {exc}", file=sys.stderr)
            return 1
        if not ipa:
            print(f"[Russian G2P] FAILED: {text}", file=sys.stderr)
            return 1
        print(f"{text}\n  accented: {accented}\n  ipa     : {ipa}")
        # Anything outside the 256-token table silently becomes PAD at training
        # and synthesis time, so surface it here where it is still debuggable.
        oov = unknown_symbols(to_vocab_ipa(ipa))
        if oov:
            detail = " ".join(f"{c!r}(U+{ord(c):04X})" for c in oov)
            print(f"  WARNING : symbols outside the vocab -> PAD: {detail}", file=sys.stderr)
        results.append(ipa)
    if args.out:
        import json

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Russian G2P] wrote {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="russian_librispeech directory")
    ap.add_argument("--out", default=None, help="output CSV (default: <root>/metadata_ipa.csv)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 8))
    ap.add_argument("--device", default="CPU", choices=["CPU", "CUDA"])
    ap.add_argument("--accent-model", default="turbo3.1")
    ap.add_argument("--phon-model", default="big")
    ap.add_argument("--workdir", default=None, help="RUPhon model cache directory")
    ap.add_argument("--limit", type=int, default=0, help="only process N rows (benchmarking)")
    ap.add_argument(
        "--text",
        action="append",
        help="Phonemize these sentences and print IPA instead of processing the corpus; "
        "repeatable. Writes --out as a JSON list when given, for run_ipa_inference.py.",
    )
    args = ap.parse_args()

    if args.text:
        return _phonemize_texts(args)

    src = os.path.join(args.root, "metadata.csv")
    out = args.out or os.path.join(args.root, "metadata_ipa.csv")
    workdir = args.workdir or os.path.join(args.root, ".ruphon_models")

    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"[Russian G2P] no rows in {src}", file=sys.stderr)
        return 1

    texts = [r["text"] for r in rows]
    print(
        f"[Russian G2P] {len(texts)} rows | {args.workers} workers | {args.device} "
        f"| accent={args.accent_model} phon={args.phon_model}",
        flush=True,
    )

    # Set in the parent too, so forked children inherit it even if a runtime
    # gets imported before the initializer runs.
    _pin_single_thread()

    t0 = time.time()
    chunk = max(1, len(texts) // (args.workers * 16))
    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.device, workdir, args.accent_model, args.phon_model),
    ) as pool:
        ipa = []
        for i, val in enumerate(pool.imap(_phonemize_one, texts, chunksize=chunk), 1):
            ipa.append(val)
            if i % 2000 == 0 or i == len(texts):
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(texts) - i) / max(1e-9, rate)
                print(f"  {i}/{len(texts)}  {rate:.0f} rows/s  ETA {eta/60:.1f} min", flush=True)
    elapsed = time.time() - t0

    failed = sum(1 for x in ipa if not x)
    fieldnames = list(rows[0].keys())
    if "ipa" not in fieldnames:
        fieldnames.append("ipa")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, val in zip(rows, ipa):
            row["ipa"] = val
            writer.writerow(row)

    print(
        f"[Russian G2P] wrote {out} in {elapsed/60:.1f} min "
        f"({len(texts)/max(1e-9, elapsed):.0f} rows/s), {failed} failures",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
