"""
build_manifest_from_hf.py

Builds manifest.csv (audio_path, reference_text, language) from a Hugging
Face audio dataset, so you don't need local .wav files or hand-typed rows.

Saves each selected sample's audio to a local .wav file (Whisper's
feature extractor needs a file path or array either way, and having
real files on disk makes debugging/listening back much easier), then
writes the manifest pointing at them.

COMMON SETSWANA / LOW-RESOURCE SOUTHERN AFRICAN OPTIONS
---------------------------------------------------------
FLEURS (Setswana):
    --dataset_name google/fleurs --dataset_config tn_za --split test
    audio column: "audio"   text column: "transcription"

Common Voice (Setswana, if/when available in your CV version):
    --dataset_name mozilla-foundation/common_voice_17_0 --dataset_config tn
    audio column: "audio"   text column: "sentence"

If you're pulling from your own WAXAL challenge notebook's dataset
instead, pass whatever --dataset_name/--dataset_config that used, and
adjust --audio_column/--text_column to match its schema.

USAGE
-----
pip install datasets soundfile --break-system-packages

python build_manifest_from_hf.py \
    --dataset_name google/fleurs --dataset_config tn_za --split test \
    --audio_column audio --text_column transcription \
    --language Setswana \
    --n_samples 40 \
    --output_dir audio_cache \
    --manifest_out manifest.csv

This downloads the dataset (cached by `datasets` after first run),
writes up to --n_samples wav files into --output_dir, and writes
manifest.csv in the current directory ready for run_experiment.py.
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", required=True, help="HF dataset id, e.g. google/fleurs")
    parser.add_argument("--dataset_config", default=None, help="HF dataset config/subset, e.g. tn_za")
    parser.add_argument("--split", default="test", help="Dataset split to pull from")
    parser.add_argument("--audio_column", default="audio", help="Name of the audio column in the dataset")
    parser.add_argument("--text_column", default="transcription", help="Name of the reference-text column")
    parser.add_argument("--language", default="Setswana", help="Label to write into the manifest's language column")
    parser.add_argument("--n_samples", type=int, default=40, help="How many utterances to pull")
    parser.add_argument("--output_dir", default="audio_cache", help="Where to save extracted .wav files")
    parser.add_argument("--manifest_out", default="manifest.csv", help="Path to write the manifest CSV")
    parser.add_argument(
        "--streaming", action="store_true",
        help="Stream the dataset instead of downloading it fully first. "
             "Much faster for small --n_samples test runs; pulls only the "
             "samples actually used instead of the whole split file.",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    import soundfile as sf

    print(f"Loading {args.dataset_name} (config={args.dataset_config}, split={args.split}, streaming={args.streaming}) ...")
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=args.streaming)

    if args.streaming:
        # Streaming datasets don't support len()/column_names the same way;
        # peek at the first example instead to validate column names.
        ds_iter = iter(ds)
        first_example = next(ds_iter)
        available_columns = list(first_example.keys())
        if args.audio_column not in available_columns:
            raise ValueError(f"'{args.audio_column}' not found. Available columns: {available_columns}")
        if args.text_column not in available_columns:
            raise ValueError(f"'{args.text_column}' not found. Available columns: {available_columns}")

        def example_generator():
            yield first_example
            for ex in ds_iter:
                yield ex

        examples = example_generator()
    else:
        if args.audio_column not in ds.column_names:
            raise ValueError(
                f"'{args.audio_column}' not found. Available columns: {ds.column_names}"
            )
        if args.text_column not in ds.column_names:
            raise ValueError(
                f"'{args.text_column}' not found. Available columns: {ds.column_names}"
            )
        n = min(args.n_samples, len(ds))
        examples = iter(ds.select(range(n)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, example in enumerate(examples):
        if i >= args.n_samples:
            break
        audio = example[args.audio_column]
        array = audio["array"]
        sr = audio["sampling_rate"]
        text = example[args.text_column]

        wav_path = output_dir / f"utt_{i:04d}.wav"
        sf.write(str(wav_path), array, sr)

        rows.append(
            {
                "audio_path": str(wav_path.resolve()),
                "reference_text": text,
                "language": args.language,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  wrote {i + 1}/{args.n_samples} audio files")

    with open(args.manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "reference_text", "language"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.manifest_out}")
    print(f"Audio files saved under {output_dir.resolve()}")
    print("\nNow run, e.g.:")
    print(
        f"  python run_experiment.py --model_path openai/whisper-base "
        f"--manifest {args.manifest_out} --output_dir results/ --ranks 2 4 --n_utterances 5"
    )


if __name__ == "__main__":
    main()