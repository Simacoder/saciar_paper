"""
real_experiment_isizulu.py

Final scoped-down experiment: isiZulu only, using the fine-tuned
checkpoint mbzuai-paris/Whisper-tiny-ZU (a real, public model actually
trained on isiZulu -- unlike vanilla whisper-tiny/base, which showed
essentially zero working capability on any of the three target
languages in the previous test run).

Setswana and Sesotho are NOT run here -- no equivalent fine-tuned
checkpoint is confirmed to exist for them. In the paper, cite the
existing Setswana WER result from the prior compression paper as
motivating evidence, and note Setswana/Sesotho compression-trust
analysis as future work pending a suitable fine-tuned baseline.

STACK
-----
- Model: transformers.WhisperForConditionalGeneration (this checkpoint
  is HF-hosted, so openai-whisper's whisper.load_model() can't load it
  -- only official OpenAI checkpoints work with that package).
- Compression: the repo's own LowRankLinear + apply_low_rank_to_whisper,
  driven by rank_ratio (fraction of full rank kept). This code is
  loader-agnostic -- it walks nn.Linear submodules regardless of how
  the model was constructed, so it's unchanged from benchmark.py.
- Dataset: dsfsi-anv/za-african-next-voices, "zul" config.
- WER: jiwer, matching the repo.

VERIFIED CHECKPOINT
--------------------
"mbzuai-paris/Whisper-tiny-ZU" does NOT exist (404). The asr-africa
NCHLT checkpoints are real but GATED (403 without approved access
request at https://huggingface.co/asr-africa/... -- request access
there if you want to try them later; approval isn't guaranteed before
your deadline).

Using instead: "TheirStory/whisper-medium-zulu" -- NOT gated, a
fine-tuned openai/whisper-medium checkpoint with a genuine self-reported
WER of 0.1993 on its own eval set (wjbmattingly/zulu_merged_audio),
confirmed directly from its model card. This is the first checkpoint
in this whole process with credible evidence of actually working on
isiZulu. Note: whisper-medium is ~769M params, much slower on CPU than
tiny -- budget accordingly, and note this is a DIFFERENT training
corpus than dsfsi-anv/za-african-next-voices, so treat any WER gap as
a possible domain-mismatch effect, not a compression effect, unless
ruled out.

REQUIREMENTS
------------
pip install torch transformers jiwer datasets soundfile scipy numpy pandas matplotlib --break-system-packages

USAGE
-----
python real_experiment_isizulu.py \
    --checkpoint mbzuai-paris/Whisper-tiny-ZU \
    --rank_ratios 0.5 0.25 0.1 \
    --n_samples 20 \
    --output_dir results_isizulu/
"""

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from xai_compression_trust import (
    jaccard_distance,
    spearman_stability,
    estimate_patch_distribution_entropy,
    eym_capacity,
    fano_error_floor,
)


# ----------------------------------------------------------------------
# Low-rank compression -- verbatim logic from the repo's benchmark.py,
# just returning the singular spectra too (for the Fano capacity calc).
# ----------------------------------------------------------------------

def build_low_rank_module():
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
                self.sigma_full = S.detach().cpu().numpy()

        def forward(self, x):
            return F.linear(x, self.U @ self.V, self.bias)

    def apply_low_rank_to_whisper(model, rank_ratio: float = 0.25, layer_scope: str = "encoder"):
        """
        layer_scope controls which parts of the model get compressed:
          - "encoder": only layers under model.model.encoder (or model.encoder).
            Excludes proj_out (vocab output head) and decoder entirely.
            Matches typical published ASR-compression methodology, and is
            the recommended default given the output-head collapse seen
            in the all-layers condition.
          - "encoder_decoder": encoder + decoder internal layers, but still
            excludes proj_out specifically (the vocab projection head).
          - "all": original repo behavior -- every nn.Linear, including
            proj_out. Kept available as an explicit ablation condition to
            confirm/quantify the output-head collapse effect.
        """
        replaced = 0
        total_saved = 0
        spectra: Dict[str, np.ndarray] = {}

        def in_scope(full_name: str) -> bool:
            if layer_scope == "all":
                return True
            if "proj_out" in full_name or "lm_head" in full_name:
                return False  # always exclude the vocab output head unless scope == "all"
            if layer_scope == "encoder":
                return "encoder" in full_name
            if layer_scope == "encoder_decoder":
                return True
            raise ValueError(f"Unknown layer_scope: {layer_scope}")

        def replace_linear(module, prefix=""):
            nonlocal replaced, total_saved
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Linear):
                    if not in_scope(full_name):
                        continue
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
        print(f"  [LRF scope={layer_scope}] Replaced {replaced} layers, saved {total_saved:,} params "
              f"({total_saved * 4 / 1e6:.1f} MB)")
        return model, spectra

    return LowRankLinear, apply_low_rank_to_whisper


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------

def load_isizulu_dataset(n_samples: int, split: str = "train") -> List[dict]:
    from datasets import load_dataset, Audio

    print("  Loading isiZulu (zul) samples...")
    dataset = load_dataset(
        "dsfsi-anv/za-african-next-voices", "zul", split=split, streaming=True
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    samples = []
    for i, item in enumerate(dataset):
        if i >= n_samples:
            break
        audio = item["audio"]["array"]
        transcript = item.get("transcript", "")
        if transcript and len(audio) > 8000:
            samples.append({"audio": np.asarray(audio, dtype=np.float32), "transcript": transcript.lower()})
        if (i + 1) % 10 == 0:
            print(f"    Loaded {i + 1}/{n_samples} samples...")

    print(f"   Loaded {len(samples)} samples")
    return samples


# ----------------------------------------------------------------------
# Occlusion importance map, teacher-forced via transformers'
# model(input_features=..., labels=...)
# ----------------------------------------------------------------------

def make_patch_grid(mel_shape, patch_freq, patch_time):
    n_freq, n_time = mel_shape
    patches = []
    for f0 in range(0, n_freq, patch_freq):
        for t0 in range(0, n_time, patch_time):
            patches.append((
                slice(f0, min(f0 + patch_freq, n_freq)),
                slice(t0, min(t0 + patch_time, n_time)),
            ))
    return patches


def teacher_forced_logprob(model, processor, mel, reference_text, device):
    import torch

    labels = processor.tokenizer(reference_text, return_tensors="pt").input_ids.to(device)
    input_features = torch.from_numpy(mel).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_features=input_features, labels=labels)
    n_tokens = labels.shape[-1]
    return -out.loss.item() * n_tokens


def occlusion_importance_map(model, processor, mel, reference_text, patch_freq, patch_time, device):
    patches = make_patch_grid(mel.shape, patch_freq, patch_time)
    baseline = teacher_forced_logprob(model, processor, mel, reference_text, device)
    importance = np.zeros(len(patches))
    for i, (f_slice, t_slice) in enumerate(patches):
        masked = mel.copy()
        masked[f_slice, t_slice] = 0.0
        score = teacher_forced_logprob(model, processor, masked, reference_text, device)
        importance[i] = baseline - score
    return importance, patches


def top_m_indices(importance, m):
    m = min(m, len(importance))
    return set(np.argsort(importance)[-m:].tolist())


def transcribe(model, processor, mel, device):
    import torch
    input_features = torch.from_numpy(mel).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().lower()


def compute_wer(reference: str, hypothesis: str) -> float:
    import jiwer
    return float(jiwer.wer(reference, hypothesis))


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------

def run_sweep(checkpoint, rank_ratios, n_samples, patch_freq, patch_time, top_m, device, output_dir, layer_scope="encoder"):
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperFeatureExtractor,
        WhisperTokenizer,
    )

    is_local = os.path.isdir(checkpoint)
    if is_local:
        abs_path = os.path.abspath(checkpoint)
        print(f"Detected local checkpoint directory: {abs_path}")
        try:
            contents = os.listdir(checkpoint)
            print(f"  Contents ({len(contents)} files): {contents}")
        except Exception as e:
            print(f"  Could not list directory contents: {e}")
    else:
        print(f"'{checkpoint}' was NOT found as a local directory "
              f"(current working directory: {os.getcwd()}). "
              f"Will attempt to treat it as a Hugging Face Hub repo id instead.")
    load_kwargs = {"local_files_only": True} if is_local else {}

    print(f"Loading {checkpoint} ...")
    # NOTE: WhisperProcessor.from_pretrained() triggers a buggy chat-template
    # Hub lookup in some transformers versions, even for local paths with
    # local_files_only=True. Loading feature_extractor + tokenizer directly
    # and constructing WhisperProcessor manually avoids that code path.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint, **load_kwargs)
    tokenizer = WhisperTokenizer.from_pretrained(checkpoint, **load_kwargs)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    base_model = WhisperForConditionalGeneration.from_pretrained(checkpoint, **load_kwargs).to(device)
    base_model.eval()

    samples = load_isizulu_dataset(n_samples)
    if not samples:
        raise RuntimeError("No isiZulu samples loaded -- check dataset access/config name 'zul'.")

    mels = []
    for s in samples:
        inputs = processor.feature_extractor(s["audio"], sampling_rate=16000, return_tensors="np")
        mels.append(inputs.input_features[0])

    LowRankLinear, apply_low_rank_to_whisper = build_low_rank_module()

    per_utt_rows, per_rank_rows = [], []

    for ratio in rank_ratios:
        print(f"\n--- rank_ratio={ratio} ---")
        compressed_model = copy.deepcopy(base_model)
        compressed_model, spectra = apply_low_rank_to_whisper(compressed_model, ratio, layer_scope)
        compressed_model.to(device)
        compressed_model.eval()

        jaccards, spearmans, top_sets = [], [], []
        wers_orig, wers_comp = [], []
        n_patches = None

        for i, (s, mel) in enumerate(zip(samples, mels)):
            ref_text = s["transcript"]

            imp_orig, patches = occlusion_importance_map(base_model, processor, mel, ref_text, patch_freq, patch_time, device)
            imp_comp, _ = occlusion_importance_map(compressed_model, processor, mel, ref_text, patch_freq, patch_time, device)
            n_patches = len(patches)

            top_orig = top_m_indices(imp_orig, top_m)
            top_comp = top_m_indices(imp_comp, top_m)
            jac = jaccard_distance(top_orig, top_comp)
            spear = spearman_stability(imp_orig, imp_comp)
            jaccards.append(jac)
            spearmans.append(spear)
            top_sets.append(top_orig)

            hyp_orig = transcribe(base_model, processor, mel, device)
            hyp_comp = transcribe(compressed_model, processor, mel, device)
            wer_orig = compute_wer(ref_text, hyp_orig)
            wer_comp = compute_wer(ref_text, hyp_comp)
            wers_orig.append(wer_orig)
            wers_comp.append(wer_comp)

            per_utt_rows.append({
                "rank_ratio": ratio, "utterance_idx": i, "reference_text": ref_text,
                "jaccard_instability": jac, "spearman_stability": spear,
                "wer_original": wer_orig, "wer_compressed": wer_comp,
                "transcription_original": hyp_orig, "transcription_compressed": hyp_comp,
            })

            if (i + 1) % 5 == 0:
                print(f"    {i + 1}/{len(samples)} utterances processed")

        mean_jaccard = float(np.mean(jaccards))
        mean_spearman = float(np.nanmean(spearmans))
        mean_wer_orig = float(np.mean(wers_orig))
        mean_wer_comp = float(np.mean(wers_comp))

        H_S = estimate_patch_distribution_entropy(top_sets, n_patches)
        caps = [eym_capacity(sig, max(1, int(round(ratio * len(sig))))) for sig in spectra.values()]
        C_eff = min(caps) if caps else float("inf")
        T = n_patches
        support_size = 2 ** min(int(math.log2(math.comb(n_patches, min(top_m, n_patches)))) + 1, 62)
        floor = fano_error_floor(H_S, C_eff, T, support_size)

        wer_increase = mean_wer_comp - mean_wer_orig
        status = "OK"
        if mean_wer_orig > 0.75:
            status = "BASELINE_BROKEN"
            print(f"    WARNING: baseline WER={mean_wer_orig:.2f} is already very high -- "
                  f"check checkpoint id / dataset config before trusting this rank's numbers.")
        elif mean_wer_comp > 0.75 or wer_increase > 0.30:
            status = "COMPRESSED_BROKEN"
            print(f"    WARNING: rank_ratio={ratio} compressed WER={mean_wer_comp:.2f} "
                  f"(from {mean_wer_orig:.2f}) -- may be broken at this ratio, not just less explainable.")

        per_rank_rows.append({
            "rank_ratio": ratio, "mean_jaccard_instability": mean_jaccard,
            "mean_spearman_stability": mean_spearman, "H_S_bits": H_S, "C_eff_bits": C_eff,
            "T_queries": T, "predicted_fano_floor": floor,
            "mean_wer_original": mean_wer_orig, "mean_wer_compressed": mean_wer_comp,
            "status": status,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_utt_rows).to_csv(output_dir / "per_utterance_results.csv", index=False)
    pd.DataFrame(per_rank_rows).to_csv(output_dir / "per_rank_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(per_rank_rows, key=lambda r: r["rank_ratio"])
    ratios_x = [r["rank_ratio"] for r in rows]
    floors = [r["predicted_fano_floor"] for r in rows]
    observed = [r["mean_jaccard_instability"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ratios_x, floors, marker="o", label="Predicted Fano floor")
    ax.plot(ratios_x, observed, marker="s", label="Observed Jaccard instability")
    ax.set_xlabel("rank_ratio (fraction of full rank kept)")
    ax.set_ylabel("Attribution error / instability")
    ax.set_title("Trust floor vs. observed instability -- isiZulu")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "floor_vs_instability_isizulu.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved results to {output_dir}/")
    print("\nPer-rank summary:")
    print(pd.DataFrame(per_rank_rows).to_string(index=False))

    return per_utt_rows, per_rank_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="TheirStory/whisper-medium-zulu")
    parser.add_argument("--rank_ratios", nargs="+", type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5])
    parser.add_argument("--n_samples", type=int, default=15)
    parser.add_argument("--patch_freq", type=int, default=10)
    parser.add_argument("--patch_time", type=int, default=75)
    parser.add_argument("--top_m", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", default="results_isizulu")
    parser.add_argument(
        "--layer_scope", default="encoder", choices=["encoder", "encoder_decoder", "all"],
        help="Which layers to compress. 'encoder' (default, recommended) excludes the "
             "vocab output head and decoder entirely, matching typical published "
             "methodology and avoiding the output-head collapse seen with 'all'. "
             "'all' reproduces the original repo behavior (includes proj_out) as an "
             "explicit ablation to quantify that collapse effect if wanted later.",
    )
    args = parser.parse_args()

    run_sweep(
        args.checkpoint, args.rank_ratios, args.n_samples,
        args.patch_freq, args.patch_time, args.top_m, args.device,
        Path(args.output_dir), args.layer_scope,
    )


if __name__ == "__main__":
    main()