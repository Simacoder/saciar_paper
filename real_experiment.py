"""
real_experiment.py

Occlusion + information-theoretic trust-floor experiment, rewritten to
match the ACTUAL stack used in https://github.com/Simacoder/paper:

  - openai-whisper package (whisper.load_model), not transformers
  - dsfsi-anv/za-african-next-voices dataset, with the repo's own
    LANG_CODES mapping (isizulu/setswana/sesotho -> zul/tsn/sot)
  - the repo's own LowRankLinear + apply_low_rank_to_whisper, driven by
    rank_ratio (fraction of full rank kept), not an absolute rank
  - jiwer for WER, matching the repo's benchmark.py

KEY FIX vs. the repo's benchmark.py: that script calls
    model.transcribe(audio, language="en", task="transcribe")
forcing English on isiZulu/Setswana/Sesotho audio. None of these three
languages are in Whisper's supported language list at all (confirmed
directly against the tokenizer's language table), so forcing "en" is
not a workaround for an unsupported code -- it's decoding the audio as
if it were English speech, which will produce fluent-sounding but
unrelated English text (this is exactly what happened in the earlier
transformers-based test run: "I'm going to go to the next room").

This script defaults to auto-detection (no language forced) instead,
which is the more honest zero-shot baseline given none of the three
target languages are supported. If forcing "en" was intentional for a
specific reason in your paper, pass --force_english to reproduce that
behavior instead.

REQUIREMENTS
------------
pip install openai-whisper jiwer datasets soundfile scipy numpy pandas matplotlib --break-system-packages

Also requires ffmpeg on PATH (openai-whisper shells out to it for audio
loading) -- same ffmpeg you may already have needed for torchcodec
earlier; if not installed, see:
  winget install Gyan.FFmpeg   (Windows)

USAGE
-----
python real_experiment.py \
    --model_size tiny \
    --languages isizulu setswana sesotho \
    --rank_ratios 0.5 0.25 0.1 \
    --n_samples 20 \
    --output_dir results_real/
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the transformers-independent math utilities from the earlier
# module -- these functions don't import transformers at call time,
# only load_whisper_model()/svd_compress_model() do (which we don't
# use here), so this import is safe without transformers installed.
from xai_compression_trust import (
    jaccard_distance,
    spearman_stability,
    estimate_patch_distribution_entropy,
    eym_capacity,
    fano_error_floor,
)

# ----------------------------------------------------------------------
# Language codes, taken directly from the repo's benchmark.py
# ----------------------------------------------------------------------
LANG_CODES = {
    "isizulu": "zul",
    "isixhosa": "xho",
    "sesotho": "sot",
    "setswana": "tsn",
    "xitsonga": "tso",
    "siswati": "ssw",
    "tshivenda": "ven",
    "isindebele": "nbl",
}


# ----------------------------------------------------------------------
# Low-rank compression -- verbatim from the repo's compression.py /
# benchmark.py (LowRankLinear + apply_low_rank_to_whisper), so the
# compression this script measures is IDENTICAL to what your paper's
# WER numbers were computed against.
# ----------------------------------------------------------------------

def build_low_rank_module():
    """
    Returns (LowRankLinear, apply_low_rank_to_whisper), imported lazily
    so this file can be inspected/compiled without torch installed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LowRankLinear(nn.Module):
        def __init__(self, original_layer: "nn.Linear", rank: int):
            super().__init__()
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features
            self.rank = rank

            with torch.no_grad():
                U, S, V = torch.linalg.svd(original_layer.weight.float())
                self.U = nn.Parameter((U[:, :rank] * S[:rank].unsqueeze(0)))
                self.V = nn.Parameter(V[:rank, :])
                self.bias = original_layer.bias.clone() if original_layer.bias is not None else None
                # Keep the full singular spectrum for the EYM capacity
                # calculation -- not part of the original repo code,
                # added for the trust-floor analysis.
                self.sigma_full = S.detach().cpu().numpy()

        def forward(self, x):
            return F.linear(x, self.U @ self.V, self.bias)

    def apply_low_rank_to_whisper(model, rank_ratio: float = 0.25):
        replaced = 0
        total_saved = 0
        spectra: Dict[str, np.ndarray] = {}

        def replace_linear(module, prefix=""):
            nonlocal replaced, total_saved
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear):
                    in_f, out_f = child.in_features, child.out_features
                    rank = max(1, int(rank_ratio * min(in_f, out_f)))
                    orig = in_f * out_f
                    comp = rank * (in_f + out_f)

                    if comp < orig:
                        lr_layer = LowRankLinear(child, rank)
                        setattr(module, name, lr_layer)
                        spectra[full_name] = lr_layer.sigma_full
                        replaced += 1
                        total_saved += (orig - comp)
                else:
                    replace_linear(child, full_name)

        replace_linear(model)
        print(f"  [LRF] Replaced {replaced} layers, saved {total_saved:,} params "
              f"({total_saved * 4 / 1e6:.1f} MB)")
        return model, spectra

    return LowRankLinear, apply_low_rank_to_whisper


# ----------------------------------------------------------------------
# Dataset loading -- matches the repo's load_za_dataset
# ----------------------------------------------------------------------

def load_za_dataset(lang_code: str, num_samples: int, split: str = "train") -> List[dict]:
    from datasets import load_dataset, Audio

    print(f"  Loading {lang_code} samples...")
    dataset = load_dataset(
        "dsfsi-anv/za-african-next-voices", lang_code, split=split, streaming=True
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    samples = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        audio = item["audio"]["array"]
        transcript = item.get("transcript", "")
        if transcript and len(audio) > 8000:
            samples.append(
                {
                    "audio": np.asarray(audio, dtype=np.float32),
                    "transcript": transcript.lower(),
                }
            )
        if (i + 1) % 10 == 0:
            print(f"    Loaded {i + 1}/{num_samples} samples...")

    print(f"   Loaded {len(samples)} samples for {lang_code}")
    return samples


# ----------------------------------------------------------------------
# Occlusion importance map, teacher-forced against the reference
# transcript, using openai-whisper's raw forward(mel, tokens) API.
# ----------------------------------------------------------------------

@dataclass
class OcclusionConfig:
    patch_time: int = 150
    patch_freq: int = 20
    fill_value: float = 0.0


def make_patch_grid(mel_shape: Tuple[int, int], cfg: OcclusionConfig) -> List[Tuple[slice, slice]]:
    n_freq, n_time = mel_shape
    patches = []
    for f0 in range(0, n_freq, cfg.patch_freq):
        for t0 in range(0, n_time, cfg.patch_time):
            patches.append((
                slice(f0, min(f0 + cfg.patch_freq, n_freq)),
                slice(t0, min(t0 + cfg.patch_time, n_time)),
            ))
    return patches


def teacher_forced_logprob(model, tokenizer, mel, reference_text: str, device: str) -> float:
    """
    Sum of log P(token_i | tokens_<i, mel) over the reference transcript,
    using openai-whisper's Whisper.forward(mel, tokens), which returns
    logits = decoder(tokens, encoder(mel)).

    Note: the tokenizer's language tag (see get_reference_tokenizer)
    is a formatting placeholder only -- it does not affect whether the
    reference text itself can be encoded, since Whisper's BPE tokenizer
    handles arbitrary UTF-8 text regardless of the declared language
    token. What matters for this metric is internal consistency between
    the original and compressed model on the SAME token sequence, which
    holds regardless of the placeholder language.
    """
    import torch
    import torch.nn.functional as F

    tokens = list(tokenizer.sot_sequence) + tokenizer.encode(reference_text) + [tokenizer.eot]
    tokens_t = torch.tensor([tokens], device=device)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_t, tokens_t)  # (1, T, vocab)

    # Predict token[t+1] from position t -- standard teacher-forced shift.
    logits = logits[:, :-1, :]
    targets = tokens_t[:, 1:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.sum().item())


def occlusion_importance_map(model, tokenizer, mel, reference_text, cfg, device):
    patches = make_patch_grid(mel.shape, cfg)
    baseline = teacher_forced_logprob(model, tokenizer, mel, reference_text, device)

    importance = np.zeros(len(patches))
    for i, (f_slice, t_slice) in enumerate(patches):
        masked = mel.copy()
        masked[f_slice, t_slice] = cfg.fill_value
        score = teacher_forced_logprob(model, tokenizer, masked, reference_text, device)
        importance[i] = baseline - score
    return importance, patches


def top_m_indices(importance: np.ndarray, m: int) -> set:
    m = min(m, len(importance))
    return set(np.argsort(importance)[-m:].tolist())


# ----------------------------------------------------------------------
# WER via jiwer (matches the repo exactly, instead of a hand-rolled
# implementation)
# ----------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer
    return float(jiwer.wer(reference, hypothesis))


def transcribe(model, audio: np.ndarray, force_english: bool) -> str:
    """
    Real (non-teacher-forced) transcription for the WER sanity check.

    force_english=True reproduces the repo's benchmark.py behavior
    (language="en"). Default is auto-detection, since none of
    isiZulu/Setswana/Sesotho are Whisper-supported language codes to
    force in the first place.
    """
    kwargs = {"task": "transcribe"}
    if force_english:
        kwargs["language"] = "en"
    result = model.transcribe(audio, **kwargs)
    return result["text"].strip().lower()


# ----------------------------------------------------------------------
# Rank-ratio sweep, per language, computing WER + attribution instability
# + the Fano trust floor
# ----------------------------------------------------------------------

def run_language(
    language: str,
    model_size: str,
    rank_ratios: Sequence[float],
    n_samples: int,
    occlusion_cfg: OcclusionConfig,
    top_m: int,
    device: str,
    force_english: bool,
    output_dir: Path,
) -> Tuple[List[dict], List[dict]]:
    import whisper
    import copy

    lang_code = LANG_CODES.get(language.lower())
    if lang_code is None:
        raise ValueError(f"Unknown language '{language}'. Known: {list(LANG_CODES)}")

    samples = load_za_dataset(lang_code, n_samples)
    if not samples:
        raise RuntimeError(f"No samples loaded for {language} ({lang_code})")

    print(f"\nLoading Whisper-{model_size} ...")
    base_model = whisper.load_model(model_size, device=device)

    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=base_model.is_multilingual, language="en", task="transcribe"
    )

    LowRankLinear, apply_low_rank_to_whisper = build_low_rank_module()

    # Precompute mel specs once.
    mels = []
    for s in samples:
        audio = whisper.pad_or_trim(s["audio"])
        mel = whisper.log_mel_spectrogram(audio, n_mels=base_model.dims.n_mels).numpy()
        mels.append(mel)

    per_utt_rows: List[dict] = []
    per_rank_rows: List[dict] = []

    for ratio in rank_ratios:
        print(f"\n  --- rank_ratio={ratio} ---")
        compressed_model = copy.deepcopy(base_model)
        compressed_model, spectra = apply_low_rank_to_whisper(compressed_model, ratio)
        compressed_model.to(device)

        jaccards, spearmans, top_sets = [], [], []
        wers_orig, wers_comp = [], []
        n_patches = None

        for i, (s, mel) in enumerate(zip(samples, mels)):
            ref_text = s["transcript"]

            imp_orig, patches = occlusion_importance_map(base_model, tokenizer, mel, ref_text, occlusion_cfg, device)
            imp_comp, _ = occlusion_importance_map(compressed_model, tokenizer, mel, ref_text, occlusion_cfg, device)
            n_patches = len(patches)

            top_orig = top_m_indices(imp_orig, top_m)
            top_comp = top_m_indices(imp_comp, top_m)
            jac = jaccard_distance(top_orig, top_comp)
            spear = spearman_stability(imp_orig, imp_comp)

            jaccards.append(jac)
            spearmans.append(spear)
            top_sets.append(top_orig)

            hyp_orig = transcribe(base_model, s["audio"], force_english)
            hyp_comp = transcribe(compressed_model, s["audio"], force_english)
            wer_orig = compute_wer(ref_text, hyp_orig)
            wer_comp = compute_wer(ref_text, hyp_comp)
            wers_orig.append(wer_orig)
            wers_comp.append(wer_comp)

            per_utt_rows.append({
                "language": language, "rank_ratio": ratio, "utterance_idx": i,
                "reference_text": ref_text, "jaccard_instability": jac,
                "spearman_stability": spear, "wer_original": wer_orig,
                "wer_compressed": wer_comp, "transcription_original": hyp_orig,
                "transcription_compressed": hyp_comp,
            })

            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(samples)} utterances processed")

        mean_jaccard = float(np.mean(jaccards))
        mean_spearman = float(np.nanmean(spearmans))
        mean_wer_orig = float(np.mean(wers_orig))
        mean_wer_comp = float(np.mean(wers_comp))

        H_S = estimate_patch_distribution_entropy(top_sets, n_patches)
        # Weakest-link (minimum) EYM capacity across compressed layers.
        # rank_ratio is a FRACTION of full rank, so k = round(ratio * len(sigma)).
        caps = []
        for sigma_full in spectra.values():
            k = max(1, int(round(ratio * len(sigma_full))))
            caps.append(eym_capacity(sigma_full, k))
        C_eff = min(caps) if caps else float("inf")

        T = n_patches
        support_size = 2 ** min(int(math.log2(math.comb(n_patches, min(top_m, n_patches)))) + 1, 62)
        floor = fano_error_floor(H_S, C_eff, T, support_size)

        wer_increase = mean_wer_comp - mean_wer_orig
        if mean_wer_comp > 0.75 or wer_increase > 0.30:
            print(f"    WARNING: rank_ratio={ratio} compressed WER={mean_wer_comp:.2f} "
                  f"(from {mean_wer_orig:.2f}) -- model may be broken at this ratio, "
                  f"not just less explainable.")

        per_rank_rows.append({
            "language": language, "rank_ratio": ratio,
            "mean_jaccard_instability": mean_jaccard, "mean_spearman_stability": mean_spearman,
            "H_S_bits": H_S, "C_eff_bits": C_eff, "T_queries": T,
            "predicted_fano_floor": floor,
            "mean_wer_original": mean_wer_orig, "mean_wer_compressed": mean_wer_comp,
        })

    return per_utt_rows, per_rank_rows


def plot_language(per_rank_rows: List[dict], language: str, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in per_rank_rows if r["language"] == language]
    rows.sort(key=lambda r: r["rank_ratio"])
    ratios = [r["rank_ratio"] for r in rows]
    floors = [r["predicted_fano_floor"] for r in rows]
    observed = [r["mean_jaccard_instability"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios, floors, marker="o", label="Predicted Fano floor")
    ax.plot(ratios, observed, marker="s", label="Observed Jaccard instability")
    ax.set_xlabel("rank_ratio (fraction of full rank kept)")
    ax.set_ylabel("Attribution error / instability")
    ax.set_title(f"Trust floor vs. observed instability -- {language}")
    ax.invert_xaxis()  # smaller rank_ratio = more compression, shown left-to-right
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"floor_vs_instability_{language}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_size", default="tiny", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--languages", nargs="+", default=["isizulu", "setswana", "sesotho"])
    parser.add_argument("--rank_ratios", nargs="+", type=float, default=[0.5, 0.25, 0.1])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--patch_time", type=int, default=150)
    parser.add_argument("--patch_freq", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="results_real")
    parser.add_argument(
        "--force_english", action="store_true",
        help="Reproduce the repo's benchmark.py behavior (language='en'). "
             "Default is auto-detection, since zu/tn/st aren't supported "
             "Whisper language codes to force in the first place.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    occlusion_cfg = OcclusionConfig(patch_time=args.patch_time, patch_freq=args.patch_freq)

    all_per_utt, all_per_rank = [], []
    for language in args.languages:
        print(f"\n=== Language: {language} ===")
        per_utt, per_rank = run_language(
            language, args.model_size, args.rank_ratios, args.n_samples,
            occlusion_cfg, args.top_m, args.device, args.force_english, output_dir,
        )
        all_per_utt.extend(per_utt)
        all_per_rank.extend(per_rank)
        plot_language(all_per_rank, language, output_dir)

    pd.DataFrame(all_per_utt).to_csv(output_dir / "per_utterance_results.csv", index=False)
    pd.DataFrame(all_per_rank).to_csv(output_dir / "per_rank_summary.csv", index=False)
    print(f"\nSaved results to {output_dir}/")


if __name__ == "__main__":
    main()