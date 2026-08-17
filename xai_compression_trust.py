"""
xai_compression_trust.py

Empirical pipeline for: "Information-Theoretic Limits of Explainability:
When Can We Trust XAI Under Model Compression?"

What this script does, per compression rank k:
  1. Loads the uncompressed Whisper model and a rank-k SVD-compressed variant
     (either from an existing checkpoint you already produced, or computed
     on-the-fly via truncated SVD on the target linear layers).
  2. For a set of held-out utterances, runs grid-occlusion on the log-mel
     spectrogram against both models, teacher-forced against the reference
     transcript, to get a patch-importance map for each.
  3. Computes attribution instability between the uncompressed and
     compressed importance maps: Jaccard distance on top-k patch sets, and
     Spearman rank correlation on full importance scores.
  4. Estimates H(S) (entropy of the "important patch" distribution) from
     the uncompressed model's attributions across the held-out set.
  5. Computes the Eckart-Young-Mirsky-based channel capacity C_comp(k) from
     the retained singular-value spectrum at that rank.
  6. Computes the Fano lower bound P_e_floor(k) and compares it against the
     observed instability, across ranks.

Design choices made for tractability under a paper deadline:
  - Occlusion, not KernelSHAP/LIME, is used as the explainer, so query
    budget T is exactly controlled (T = number of masked grid cells).
  - S(x) is defined as the top-m most important time-frequency patches
    of the log-mel spectrogram, ranked by the drop they cause in the
    teacher-forced log-probability of the reference transcript.
  - H(S) is estimated non-parametrically as the entropy of the empirical
    distribution over which patches get selected as "important" across
    the held-out utterance set, NOT assumed Gaussian or otherwise
    parametric. Only C_comp uses a Gaussian-channel modeling assumption
    (documented at that function) -- flag this in the paper as an
    assumption, not a derived fact.

This script does not download or run against real checkpoints in this
sandbox (no network access to model hubs here). It is meant to be run
in your own environment, pointed at your existing Whisper + compressed
checkpoints. A synthetic-data self-test is included at the bottom so you
can sanity-check the statistics pipeline (Jaccard/entropy/Fano/capacity
plumbing) before spending compute on real audio.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

# ----------------------------------------------------------------------
# 1. Model loading / compression
# ----------------------------------------------------------------------
# These two functions are the integration points with your existing
# compression code. Swap the bodies for your actual checkpoint-loading
# logic (you already have this from the MDPI compression paper).

def load_whisper_model(model_name_or_path: str, device: str = "cpu"):
    """
    Load a Whisper model + processor. Replace with your existing loader.

    Expected to return (model, processor) where model is a
    transformers.WhisperForConditionalGeneration and processor is a
    transformers.WhisperProcessor (or your equivalent wrapper).
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()
    return model, processor


def svd_compress_linear_layer(weight: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncated SVD compression of a single 2D weight matrix.

    Returns:
        W_k: the rank-k reconstruction of `weight`
        sigma_full: the full singular value spectrum of `weight`
                    (needed later for the EYM capacity estimate)
    """
    U, sigma, Vt = np.linalg.svd(weight, full_matrices=False)
    W_k = (U[:, :rank] * sigma[:rank]) @ Vt[:rank, :]
    return W_k, sigma


def svd_compress_model(model, rank: int, target_layer_names: Optional[Sequence[str]] = None):
    """
    Apply rank-k truncated SVD to the target linear layers of a Whisper
    model (in-place on a deep copy), returning the compressed model plus
    a dict of {layer_name: sigma_full} for capacity computation.

    target_layer_names: if None, applies to all `nn.Linear` weight
    matrices in the encoder (matches the scope of your MDPI compression
    paper). Pass an explicit list to restrict to specific layers if you
    want per-layer capacity analysis instead of a whole-encoder figure.
    """
    import torch
    import torch.nn as nn

    model_c = copy.deepcopy(model)
    spectra: Dict[str, np.ndarray] = {}

    for name, module in model_c.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_layer_names is not None and name not in target_layer_names:
            continue
        if "encoder" not in name:
            continue  # restrict to encoder, matching prior compression work

        W = module.weight.detach().cpu().numpy()
        max_rank = min(W.shape)
        k = min(rank, max_rank)
        W_k, sigma_full = svd_compress_linear_layer(W, k)
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(W_k).to(module.weight.dtype))
        spectra[name] = sigma_full

    return model_c, spectra


def normalize_for_wer(text: str) -> List[str]:
    """
    Minimal WER text normalization: lowercase, strip punctuation
    (keeping word-internal hyphens/apostrophes, which matter for
    isiZulu/Setswana/Sesotho orthography), collapse whitespace, split
    into words.
    """
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s'-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Standard WER via Levenshtein edit distance at the word level:
    (substitutions + insertions + deletions) / len(reference_words).

    Returns nan if the reference is empty (undefined WER).
    """
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)

    if len(ref) == 0:
        return float("nan")

    # Standard DP edit distance.
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost,  # substitution / match
            )
    return float(dp[n, m] / n)


# Whisper's built-in language codes for the languages this study targets.
# Setswana (tsn) is NOT in Whisper's supported language list -- generate()
# will fall back to auto-detection for it, which is worth noting as a
# limitation in the paper if Setswana WER checks look unstable for this
# reason rather than because of compression.
WHISPER_LANGUAGE_CODES = {
    "isizulu": "zu",
    "zulu": "zu",
    "sesotho": "st",
    "setswana": None,  # not supported by Whisper; generate() will auto-detect
}




@dataclass
class OcclusionConfig:
    patch_time: int = 10      # frames per patch along time axis
    patch_freq: int = 10      # bins per patch along frequency axis
    fill_value: float = 0.0   # what to overwrite masked patches with
                               # (0.0 in normalized log-mel space is a
                               # reasonable "silence" baseline; consider
                               # using per-utterance mean instead)


def make_patch_grid(mel_shape: Tuple[int, int], cfg: OcclusionConfig) -> List[Tuple[slice, slice]]:
    """Build the list of (freq_slice, time_slice) patches tiling the mel spectrogram."""
    n_freq, n_time = mel_shape
    patches = []
    for f0 in range(0, n_freq, cfg.patch_freq):
        for t0 in range(0, n_time, cfg.patch_time):
            f_slice = slice(f0, min(f0 + cfg.patch_freq, n_freq))
            t_slice = slice(t0, min(t0 + cfg.patch_time, n_time))
            patches.append((f_slice, t_slice))
    return patches


def teacher_forced_logprob(model, processor, mel: "np.ndarray", reference_text: str, device: str) -> float:
    """
    Compute log P(reference_text | mel) under teacher forcing.

    Returns a scalar: the *negative* of the model's cross-entropy loss
    summed (not averaged) over reference tokens, i.e. higher = more
    confident/correct prediction of the reference transcript.
    """
    import torch

    labels = processor.tokenizer(reference_text, return_tensors="pt").input_ids.to(device)
    input_features = torch.from_numpy(mel).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input_features=input_features, labels=labels)
        n_tokens = labels.shape[-1]
        # HF returns mean CE loss over tokens; convert back to summed log-prob
        total_logprob = -out.loss.item() * n_tokens
    return total_logprob


def occlusion_importance_map(
    model,
    processor,
    mel: "np.ndarray",
    reference_text: str,
    cfg: OcclusionConfig,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[slice, slice]]]:
    """
    Grid-occlusion importance map.

    For each patch, mask it and measure the DROP in teacher-forced
    log-probability of the reference transcript relative to the
    unmasked baseline. Larger drop = more important patch.

    Returns:
        importance: 1D array, one score per patch (same order as `patches`)
        patches: the list of (freq_slice, time_slice) used, so importance
                 scores can be mapped back to spectrogram regions later.
    """
    patches = make_patch_grid(mel.shape, cfg)
    baseline = teacher_forced_logprob(model, processor, mel, reference_text, device)

    importance = np.zeros(len(patches), dtype=np.float64)
    for i, (f_slice, t_slice) in enumerate(patches):
        masked = mel.copy()
        masked[f_slice, t_slice] = cfg.fill_value
        score = teacher_forced_logprob(model, processor, masked, reference_text, device)
        importance[i] = baseline - score  # positive = patch was important

    return importance, patches


# ----------------------------------------------------------------------
# 3. Attribution instability metrics
# ----------------------------------------------------------------------

def top_m_indices(importance: np.ndarray, m: int) -> set:
    """Indices of the m most important patches."""
    m = min(m, len(importance))
    return set(np.argsort(importance)[-m:].tolist())


def jaccard_distance(set_a: set, set_b: set) -> float:
    """1 - Jaccard(A, B). 0 = identical top-m sets, 1 = fully disjoint."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    inter = set_a & set_b
    return 1.0 - (len(inter) / len(union))


def spearman_stability(importance_a: np.ndarray, importance_b: np.ndarray) -> float:
    """Spearman rank correlation between two full importance vectors."""
    rho, _ = spearmanr(importance_a, importance_b)
    return float(rho)


# ----------------------------------------------------------------------
# 4. H(S) estimation across the held-out set
# ----------------------------------------------------------------------

def estimate_patch_distribution_entropy(
    top_m_sets: List[set], n_patches: int
) -> float:
    """
    Non-parametric estimate of H(S): treat "which patches get selected
    as important, across the held-out utterance set" as an empirical
    distribution over patch indices, and compute its Shannon entropy
    in bits.

    This avoids assuming any parametric form for S(x); it is simply the
    empirical frequency with which each patch index appears in a
    top-m set, normalized to a probability distribution.
    """
    counts = np.zeros(n_patches, dtype=np.float64)
    for s in top_m_sets:
        for idx in s:
            counts[idx] += 1.0
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def rank_for_compression_ratio(m: int, n: int, ratio: float) -> int:
    """
    Compute the truncated-SVD rank k that achieves a target compression
    ratio for an (m x n) weight matrix, matching the Eckart-Young-Mirsky
    parameter-count definition: ratio = (m*n) / (k*(m+n)).

    Use this instead of guessing small ranks like 2 or 4 -- for a
    typical Whisper-base 512x512 or 512x2048 layer, a "mild" 1.5-2x
    compression ratio corresponds to k in the 100s, not single digits.
    Ranks that low aren't "compressed", they're closer to ablated.
    """
    k = (m * n) / (ratio * (m + n))
    return max(1, min(int(round(k)), min(m, n)))


def ranks_for_ratio_sweep(layer_dims: Sequence[Tuple[int, int]], ratios: Sequence[float]) -> Dict[float, int]:
    """
    Given the (m, n) dims of the layers being compressed and a list of
    target compression ratios (e.g. [1.2, 1.64, 2.0, 3.0, 5.0]), return
    the conservative (minimum, i.e. most compressed) rank across layers
    for each ratio -- use this as your --ranks sweep instead of
    arbitrary small integers.
    """
    result = {}
    for ratio in ratios:
        ranks = [rank_for_compression_ratio(m, n, ratio) for (m, n) in layer_dims]
        result[ratio] = min(ranks)
    return result




def eym_capacity(sigma_full: np.ndarray, rank: int) -> float:
    """
    Gaussian-channel-style capacity estimate for a rank-k truncation of
    a weight matrix with full singular spectrum `sigma_full`.

    ASSUMPTION (flag in paper as Assumption 1, not a derived fact):
    the discarded singular-value energy is treated as additive noise
    power against the retained signal power, giving

        C_comp = 0.5 * log2(1 + signal_power / noise_power)

    Real truncation error is structured, not iid Gaussian noise; this
    is a tractable proxy, validated empirically against observed
    instability rather than assumed exact.
    """
    sigma_full = np.asarray(sigma_full, dtype=np.float64)
    rank = min(rank, len(sigma_full))
    signal_power = float((sigma_full[:rank] ** 2).sum())
    noise_power = float((sigma_full[rank:] ** 2).sum())
    if noise_power <= 0:
        return float("inf")  # no compression loss at this rank
    return 0.5 * math.log2(1.0 + signal_power / noise_power)


def aggregate_capacity(layer_spectra: Dict[str, np.ndarray], rank: int) -> float:
    """
    Aggregate per-layer capacities into a single effective capacity for
    the cascade bound. Using the minimum across layers is the
    conservative (weakest-link) choice consistent with the cascade
    framing in the derivation -- a single badly-compressed layer can
    bottleneck the whole explanation channel.
    """
    caps = [eym_capacity(sigma, rank) for sigma in layer_spectra.values()]
    return min(caps) if caps else float("inf")


# ----------------------------------------------------------------------
# 6. Fano lower bound
# ----------------------------------------------------------------------

def fano_error_floor(H_S: float, C_eff: float, T: int, support_size: int) -> float:
    """
    P_e >= (H(S) - T * C_eff - 1) / log2(support_size - 1)

    support_size: |S|, the number of distinct possible "important patch
    sets" being distinguished (in practice, use the number of grid
    patches choose m, or a tractable proxy -- see note below).

    Returns a value clipped to [0, 1]; the bound is only informative
    when it is > 0 (i.e., when compression + query budget genuinely
    can't resolve the explanation).
    """
    if support_size <= 2:
        return 0.0
    denom = math.log2(support_size - 1)
    numerator = H_S - T * C_eff - 1.0
    floor = numerator / denom
    return float(np.clip(floor, 0.0, 1.0))


# ----------------------------------------------------------------------
# 7. Per-rank pipeline
# ----------------------------------------------------------------------

@dataclass
class UtteranceResult:
    jaccard: float
    spearman: float
    top_m_uncompressed: set
    importance_uncompressed: np.ndarray
    importance_compressed: np.ndarray


@dataclass
class RankResult:
    rank: int
    mean_jaccard: float
    mean_spearman: float
    H_S_bits: float
    C_eff_bits: float
    T: int
    support_size: int
    predicted_floor: float
    mean_wer_original: float = float("nan")
    mean_wer_compressed: float = float("nan")


def run_rank(
    base_model,
    processor,
    utterances: List[Tuple["np.ndarray", str]],  # (mel, reference_text)
    rank: int,
    occlusion_cfg: OcclusionConfig,
    top_m: int,
    device: str = "cpu",
) -> RankResult:
    """
    Full pipeline for a single compression rank: compress, run occlusion
    on both models across all held-out utterances, compute instability,
    entropy, capacity, and the Fano floor.
    """
    compressed_model, spectra = svd_compress_model(base_model, rank)

    per_utt: List[UtteranceResult] = []
    n_patches = None
    for mel, ref_text in utterances:
        imp_orig, patches = occlusion_importance_map(
            base_model, processor, mel, ref_text, occlusion_cfg, device
        )
        imp_comp, _ = occlusion_importance_map(
            compressed_model, processor, mel, ref_text, occlusion_cfg, device
        )
        n_patches = len(patches)

        top_orig = top_m_indices(imp_orig, top_m)
        top_comp = top_m_indices(imp_comp, top_m)

        per_utt.append(
            UtteranceResult(
                jaccard=jaccard_distance(top_orig, top_comp),
                spearman=spearman_stability(imp_orig, imp_comp),
                top_m_uncompressed=top_orig,
                importance_uncompressed=imp_orig,
                importance_compressed=imp_comp,
            )
        )

    mean_jaccard = float(np.mean([u.jaccard for u in per_utt]))
    mean_spearman = float(np.nanmean([u.spearman for u in per_utt]))

    H_S = estimate_patch_distribution_entropy(
        [u.top_m_uncompressed for u in per_utt], n_patches
    )
    C_eff = aggregate_capacity(spectra, rank)
    T = n_patches  # one query per patch under grid occlusion

    # Support size proxy: number of ways to choose top_m patches out of
    # n_patches. For large n_patches this saturates log2(support_size)
    # quickly; capping n_patches choose top_m via log-binomial avoids
    # overflow.
    support_size = 2 ** min(
        int(math.log2(math.comb(n_patches, min(top_m, n_patches)))) + 1, 62
    )

    floor = fano_error_floor(H_S, C_eff, T, support_size)

    return RankResult(
        rank=rank,
        mean_jaccard=mean_jaccard,
        mean_spearman=mean_spearman,
        H_S_bits=H_S,
        C_eff_bits=C_eff,
        T=T,
        support_size=support_size,
        predicted_floor=floor,
    )


def run_sweep(
    base_model,
    processor,
    utterances: List[Tuple["np.ndarray", str]],
    ranks: Sequence[int],
    occlusion_cfg: OcclusionConfig = OcclusionConfig(),
    top_m: int = 20,
    device: str = "cpu",
) -> List[RankResult]:
    results = [
        run_rank(base_model, processor, utterances, r, occlusion_cfg, top_m, device)
        for r in ranks
    ]
    return results


def summarize_sweep(results: List[RankResult]) -> None:
    """Print a table and the trend-check correlation described in the scoping discussion."""
    print(f"{'rank':>6} {'jaccard':>9} {'spearman':>9} {'H(S)':>8} {'C_eff':>8} {'T':>6} {'floor':>8}")
    for r in results:
        print(
            f"{r.rank:>6} {r.mean_jaccard:>9.3f} {r.mean_spearman:>9.3f} "
            f"{r.H_S_bits:>8.3f} {r.C_eff_bits:>8.3f} {r.T:>6} {r.predicted_floor:>8.3f}"
        )

    if len(results) >= 3:
        floors = [r.predicted_floor for r in results]
        observed = [r.mean_jaccard for r in results]
        rho, p = spearmanr(floors, observed)
        print(f"\nSpearman(predicted_floor, observed_jaccard_instability) = {rho:.3f} (p={p:.3f})")
        violations = [r for r in results if r.mean_jaccard < r.predicted_floor]
        if violations:
            print(
                f"WARNING: {len(violations)} rank(s) have observed instability BELOW the "
                f"predicted floor -- revisit the Gaussian-noise capacity assumption "
                f"(ranks: {[r.rank for r in violations]})"
            )
        else:
            print("Sanity check passed: observed instability never falls below the predicted floor.")


# ----------------------------------------------------------------------
# 8. Synthetic self-test (run this first, before touching real audio)
# ----------------------------------------------------------------------

def _synthetic_self_test():
    """
    Sanity-checks the statistics plumbing (Jaccard, entropy, EYM capacity,
    Fano floor, correlation reporting) using synthetic data, with NO
    dependency on transformers/torch/real audio. Run this to confirm the
    math behaves before spending time on real checkpoints.
    """
    rng = np.random.default_rng(0)
    n_patches = 200
    n_utts = 30
    top_m = 20

    print("=== Synthetic self-test (no real model/audio required) ===\n")

    ranks = [4, 8, 16, 32, 64]
    results = []
    for rank in ranks:
        # Fake singular spectrum: decaying, with more retained energy at higher rank
        sigma_full = np.sort(rng.exponential(scale=5.0, size=100))[::-1]
        C_eff = eym_capacity(sigma_full, rank)

        # Simulate importance maps: as rank increases, compressed importance
        # correlates more with the "true" (uncompressed) importance.
        jaccards, spearmans, top_sets = [], [], []
        noise_level = 1.0 / (1.0 + rank / 8.0)  # higher rank -> less noise
        for _ in range(n_utts):
            imp_true = rng.normal(size=n_patches)
            imp_comp = imp_true + rng.normal(scale=noise_level, size=n_patches)

            top_true = top_m_indices(imp_true, top_m)
            top_comp = top_m_indices(imp_comp, top_m)

            jaccards.append(jaccard_distance(top_true, top_comp))
            spearmans.append(spearman_stability(imp_true, imp_comp))
            top_sets.append(top_true)

        H_S = estimate_patch_distribution_entropy(top_sets, n_patches)
        T = n_patches
        support_size = 2 ** min(
            int(math.log2(math.comb(n_patches, top_m))) + 1, 62
        )
        floor = fano_error_floor(H_S, C_eff, T, support_size)

        results.append(
            RankResult(
                rank=rank,
                mean_jaccard=float(np.mean(jaccards)),
                mean_spearman=float(np.nanmean(spearmans)),
                H_S_bits=H_S,
                C_eff_bits=C_eff,
                T=T,
                support_size=support_size,
                predicted_floor=floor,
            )
        )

    summarize_sweep(results)


if __name__ == "__main__":
    _synthetic_self_test()

    # ------------------------------------------------------------------
    # Real pipeline usage (uncomment and adapt to your setup):
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, processor = load_whisper_model("path/to/your/whisper-base", device=device)
    #
    # utterances = [
    #     (mel_array_1, "reference transcript 1"),
    #     (mel_array_2, "reference transcript 2"),
    #     # ... load via your existing Whisper preprocessing for
    #     # isiZulu/Setswana/Sesotho held-out utterances
    # ]
    #
    # results = run_sweep(
    #     model, processor, utterances,
    #     ranks=[4, 8, 16, 32, 64],   # bracket your reported 1.64x point
    #     top_m=20,
    #     device=device,
    # )
    # summarize_sweep(results)
    # ------------------------------------------------------------------