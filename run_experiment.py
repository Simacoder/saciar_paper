"""
run_experiment.py

End-to-end driver for the compression-trust experiment. Points the
pipeline in xai_compression_trust.py at real Whisper checkpoints and a
manifest of held-out audio + reference transcripts, and writes out:

  - <output_dir>/per_utterance_results.csv
  - <output_dir>/per_rank_summary.csv
  - <output_dir>/floor_vs_instability_<language>.png  (one plot per language)

USAGE
-----
1. Prepare a manifest CSV with columns: audio_path, reference_text, language
   (language column is optional -- omit it and everything is treated as
   one pooled set).

   Example manifest.csv:
     audio_path,reference_text,language
     /data/tsn/utt001.wav,"pula e na kwa Gaborone",Setswana
     /data/tsn/utt002.wav,"o tsamaya kae",Setswana
     /data/zul/utt001.wav,"sawubona",isiZulu

2. Run:
     python run_experiment.py \
         --model_path openai/whisper-base \
         --manifest manifest.csv \
         --output_dir results/ \
         --ranks 2 4 8 16 32 \
         --top_m 20 \
         --patch_time 10 --patch_freq 10 \
         --n_utterances 40

   Start with --ranks 2 4 (aggressive/low) per the note in the prior
   discussion -- that's where the Fano floor is most likely to be
   non-trivial (non-zero) and worth reporting.

3. Recommended first pass, given deadline pressure: restrict to Setswana
   only (--language Setswana), since it's your strongest compression
   result. Expand to isiZulu/Sesotho only if time remains.

REQUIREMENTS
------------
pip install torch transformers librosa soundfile scipy numpy pandas matplotlib tqdm --break-system-packages
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local module from the earlier scoping step.
from xai_compression_trust import (
    OcclusionConfig,
    RankResult,
    load_whisper_model,
    occlusion_importance_map,
    svd_compress_model,
    top_m_indices,
    jaccard_distance,
    spearman_stability,
    estimate_patch_distribution_entropy,
    aggregate_capacity,
    fano_error_floor,
    word_error_rate,
    WHISPER_LANGUAGE_CODES,
)

import math


def load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "audio_path" not in df.columns or "reference_text" not in df.columns:
        raise ValueError(
            "Manifest must have at least 'audio_path' and 'reference_text' columns."
        )
    if "language" not in df.columns:
        df["language"] = "all"
    return df


def load_mel(audio_path: str, processor, sampling_rate: int = 16000) -> np.ndarray:
    """
    Load an audio file and extract the log-mel spectrogram Whisper expects,
    as a plain numpy array of shape (n_mel_bins, n_frames).
    """
    import librosa

    audio, sr = librosa.load(audio_path, sr=sampling_rate)
    features = processor.feature_extractor(
        audio, sampling_rate=sampling_rate, return_tensors="np"
    )
    mel = features.input_features[0]  # shape (n_mel, n_frames), e.g. (80, 3000)
    return mel


def build_utterance_set(
    df: pd.DataFrame, processor, n_utterances: Optional[int], language: Optional[str]
) -> List[Tuple[np.ndarray, str]]:
    subset = df
    if language is not None:
        subset = subset[subset["language"] == language]
        if subset.empty:
            raise ValueError(f"No rows found for language='{language}' in manifest.")
    if n_utterances is not None:
        subset = subset.head(n_utterances)

    utterances = []
    skipped = 0
    for _, row in subset.iterrows():
        try:
            mel = load_mel(row["audio_path"], processor)
            utterances.append((mel, str(row["reference_text"])))
        except Exception as e:  # noqa: BLE001 -- surfacing bad files is more useful than crashing the run
            skipped += 1
            print(f"  [skip] {row['audio_path']}: {e}", file=sys.stderr)
    if skipped:
        print(f"Skipped {skipped} utterance(s) due to load errors.")
    if not utterances:
        raise RuntimeError("No utterances could be loaded -- check manifest paths.")
    return utterances


def transcribe(model, processor, mel: np.ndarray, language: str, device: str) -> str:
    """
    Run actual generation (not teacher-forced) to get the model's real
    transcription, for the WER sanity check.

    Attempts language forcing via WHISPER_LANGUAGE_CODES first, but
    falls back to auto-detection if the checkpoint's tokenizer doesn't
    actually support that language code (this varies by Whisper model
    size/checkpoint -- e.g. whisper-base's tokenizer rejects 'zu' even
    though larger Whisper checkpoints support it). Relevant for
    Setswana regardless, which isn't in Whisper's supported list at all.
    """
    import torch

    input_features = torch.from_numpy(mel).unsqueeze(0).to(device)
    lang_code = WHISPER_LANGUAGE_CODES.get(language.strip().lower())

    generate_kwargs = {"language": lang_code} if lang_code is not None else {}

    with torch.no_grad():
        try:
            predicted_ids = model.generate(input_features, **generate_kwargs)
        except ValueError:
            # Forced language unsupported by this checkpoint -> auto-detect instead.
            predicted_ids = model.generate(input_features)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text

def run_rank_with_logging(
    base_model,
    processor,
    utterances: List[Tuple[np.ndarray, str]],
    rank: int,
    occlusion_cfg: OcclusionConfig,
    top_m: int,
    device: str,
    per_utt_rows: List[dict],
    language_label: str,
) -> RankResult:
    """
    Same computation as xai_compression_trust.run_rank, but also appends
    a per-utterance row to per_utt_rows for CSV export, prints progress,
    and computes a WER sanity check so instability numbers can be
    interpreted against whether the compressed model is still
    transcribing sensibly, not just producing different attributions
    because it has effectively broken at this rank.
    """
    compressed_model, spectra = svd_compress_model(base_model, rank)

    jaccards, spearmans, top_sets = [], [], []
    wers_original, wers_compressed = [], []
    n_patches = None

    for i, (mel, ref_text) in enumerate(utterances):
        imp_orig, patches = occlusion_importance_map(
            base_model, processor, mel, ref_text, occlusion_cfg, device
        )
        imp_comp, _ = occlusion_importance_map(
            compressed_model, processor, mel, ref_text, occlusion_cfg, device
        )
        n_patches = len(patches)

        top_orig = top_m_indices(imp_orig, top_m)
        top_comp = top_m_indices(imp_comp, top_m)

        jac = jaccard_distance(top_orig, top_comp)
        spear = spearman_stability(imp_orig, imp_comp)

        jaccards.append(jac)
        spearmans.append(spear)
        top_sets.append(top_orig)

        hyp_orig = transcribe(base_model, processor, mel, language_label, device)
        hyp_comp = transcribe(compressed_model, processor, mel, language_label, device)
        wer_orig = word_error_rate(ref_text, hyp_orig)
        wer_comp = word_error_rate(ref_text, hyp_comp)
        wers_original.append(wer_orig)
        wers_compressed.append(wer_comp)

        per_utt_rows.append(
            {
                "language": language_label,
                "rank": rank,
                "utterance_idx": i,
                "reference_text": ref_text,
                "jaccard_instability": jac,
                "spearman_stability": spear,
                "wer_original": wer_orig,
                "wer_compressed": wer_comp,
                "transcription_original": hyp_orig,
                "transcription_compressed": hyp_comp,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"    rank={rank}: {i + 1}/{len(utterances)} utterances processed")

    mean_jaccard = float(np.mean(jaccards))
    mean_spearman = float(np.nanmean(spearmans))
    mean_wer_original = float(np.nanmean(wers_original))
    mean_wer_compressed = float(np.nanmean(wers_compressed))

    H_S = estimate_patch_distribution_entropy(top_sets, n_patches)
    C_eff = aggregate_capacity(spectra, rank)
    T = n_patches

    support_size = 2 ** min(
        int(math.log2(math.comb(n_patches, min(top_m, n_patches)))) + 1, 62
    )
    floor = fano_error_floor(H_S, C_eff, T, support_size)

    result = RankResult(
        rank=rank,
        mean_jaccard=mean_jaccard,
        mean_spearman=mean_spearman,
        H_S_bits=H_S,
        C_eff_bits=C_eff,
        T=T,
        support_size=support_size,
        predicted_floor=floor,
        mean_wer_original=mean_wer_original,
        mean_wer_compressed=mean_wer_compressed,
    )

    wer_increase = mean_wer_compressed - mean_wer_original
    if mean_wer_compressed > 0.75 or wer_increase > 0.30:
        print(
            f"    WARNING: rank={rank} WER jumped to {mean_wer_compressed:.2f} "
            f"(from {mean_wer_original:.2f}) -- the compressed model may be "
            f"broken at this rank, not just less explainable. Instability "
            f"numbers at this rank should not be used to support the "
            f"'explanations fail even when accuracy holds' claim."
        )

    return result


def plot_results(results: List[RankResult], language: str, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    ranks = [r.rank for r in results]
    floors = [r.predicted_floor for r in results]
    observed = [r.mean_jaccard for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, floors, marker="o", label="Predicted Fano floor")
    ax.plot(ranks, observed, marker="s", label="Observed Jaccard instability")
    ax.set_xlabel("Compression rank (k)")
    ax.set_ylabel("Attribution error / instability")
    ax.set_title(f"Explanation trust floor vs. observed instability -- {language}")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"floor_vs_instability_{language}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, help="Path or HF hub id of the uncompressed Whisper model")
    parser.add_argument("--manifest", required=True, help="CSV with audio_path, reference_text, [language]")
    parser.add_argument("--output_dir", default="results", help="Where to write CSVs and plots")
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16, 32], help="Compression ranks to sweep")
    parser.add_argument("--top_m", type=int, default=20, help="Number of top-important patches defining S(x)")
    parser.add_argument("--patch_time", type=int, default=10, help="Patch size along time axis")
    parser.add_argument("--patch_freq", type=int, default=10, help="Patch size along frequency axis")
    parser.add_argument("--n_utterances", type=int, default=None, help="Cap utterances per language (omit for all)")
    parser.add_argument("--language", default=None, help="Restrict to one language (e.g. Setswana). Omit for all languages found in manifest.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from {args.model_path} ...")
    model, processor = load_whisper_model(args.model_path, device=args.device)

    df = load_manifest(args.manifest)
    languages = [args.language] if args.language else sorted(df["language"].unique())

    occlusion_cfg = OcclusionConfig(patch_time=args.patch_time, patch_freq=args.patch_freq)

    all_per_utt_rows: List[dict] = []
    all_rank_summaries: List[dict] = []

    for language in languages:
        print(f"\n=== Language: {language} ===")
        utterances = build_utterance_set(df, processor, args.n_utterances, language)
        print(f"Loaded {len(utterances)} utterances.")

        results: List[RankResult] = []
        for rank in args.ranks:
            print(f"  Running rank={rank} ...")
            r = run_rank_with_logging(
                model, processor, utterances, rank, occlusion_cfg,
                args.top_m, args.device, all_per_utt_rows, language,
            )
            results.append(r)
            all_rank_summaries.append(
                {
                    "language": language,
                    "rank": r.rank,
                    "mean_jaccard_instability": r.mean_jaccard,
                    "mean_spearman_stability": r.mean_spearman,
                    "H_S_bits": r.H_S_bits,
                    "C_eff_bits": r.C_eff_bits,
                    "T_queries": r.T,
                    "predicted_fano_floor": r.predicted_floor,
                    "mean_wer_original": r.mean_wer_original,
                    "mean_wer_compressed": r.mean_wer_compressed,
                }
            )

        # Per-language sanity check + trend correlation, printed immediately
        from scipy.stats import spearmanr
        floors = [r.predicted_floor for r in results]
        observed = [r.mean_jaccard for r in results]
        if len(set(floors)) > 1:
            rho, p = spearmanr(floors, observed)
            print(f"  Spearman(predicted_floor, observed_instability) = {rho:.3f} (p={p:.3f})")
        violations = [r for r in results if r.mean_jaccard < r.predicted_floor]
        if violations:
            print(f"  WARNING: floor violated at rank(s) {[r.rank for r in violations]} -- "
                  f"revisit the Gaussian-noise capacity assumption for this language.")
        else:
            print("  Sanity check passed: no rank has observed instability below the predicted floor.")

        plot_results(results, language, output_dir)

    per_utt_path = output_dir / "per_utterance_results.csv"
    rank_summary_path = output_dir / "per_rank_summary.csv"
    pd.DataFrame(all_per_utt_rows).to_csv(per_utt_path, index=False)
    pd.DataFrame(all_rank_summaries).to_csv(rank_summary_path, index=False)

    print(f"\nSaved per-utterance results to {per_utt_path}")
    print(f"Saved per-rank summary to {rank_summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()