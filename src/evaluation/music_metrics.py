# pip install pretty_midi numpy dtw-python scikit-learn
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple
import numpy as np
import pretty_midi
from dtw import dtw
from numpy.linalg import norm
from sklearn.metrics import precision_recall_fscore_support


# ---------- helpers ----------
def _collect_onsets(pm: pretty_midi.PrettyMIDI, include_drums=True):
    """Concatenate instrument onsets, optionally skip drums, sort, and deduplicate within dedup_tol."""
    onsets = []
    for inst in pm.instruments:
        if (not include_drums) and inst.is_drum:
            continue
        onsets.extend(inst.get_onsets().tolist())
    if not onsets:
        return np.zeros((0,), dtype=float)
    onsets = np.array(sorted(onsets), dtype=float)
    return onsets


def _onset_median_durations(
    pm: pretty_midi.PrettyMIDI, include_drums=False, bin_tol=1e-3
):
    """
    Map (rounded) onset -> median duration over all notes that start 'at' that onset.
    Returns:
        times_sorted: np.array of unique onset times (keys)
        med_durs: np.array of median durations aligned with times_sorted
    """
    buckets = defaultdict(list)
    for inst in pm.instruments:
        if (not include_drums) and inst.is_drum:
            continue
        for n in inst.notes:
            key = round(n.start / bin_tol) * bin_tol  # simple time binning
            buckets[key].append(n.get_duration())
    if not buckets:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    times = np.array(sorted(buckets.keys()), dtype=float)
    meds = np.array([float(np.median(buckets[t])) for t in times], dtype=float)
    return times, meds


def _match_onsets(ref_times, hyp_times, tol=0.03):
    """
    Greedy two-pointer matching of onset *events* within absolute tolerance.
    Returns:
        pairs: list of (i_ref, j_hyp) indices
        tp, fp, fn: counts for precision/recall
    """
    i = j = 0
    pairs = []
    while i < len(ref_times) and j < len(hyp_times):
        diff = hyp_times[j] - ref_times[i]
        if abs(diff) <= tol:
            pairs.append((i, j))
            i += 1
            j += 1
        elif diff < -tol:
            j += 1
        else:
            i += 1
    tp = len(pairs)
    fp = len(hyp_times) - tp
    fn = len(ref_times) - tp
    return pairs, tp, fp, fn


def _pm_chroma(pm: pretty_midi.PrettyMIDI, fs=100, include_drums=False):
    Cs = []
    for inst in pm.instruments:
        if (not include_drums) and inst.is_drum:
            continue
        C = inst.get_chroma(fs=fs)  # (12, T)
        if C.size:
            Cs.append(C.astype(np.float32))

    if not Cs:
        return np.zeros((12, 0), dtype=np.float32)

    # Make all chroma have the same T (pad to the max length)
    T = max(C.shape[1] for C in Cs)
    chroma_sum = np.zeros((12, T), dtype=np.float32)
    for C in Cs:
        chroma_sum[:, :C.shape[1]] += C  # pad remainder with zeros

    # L2-normalize each frame
    eps = 1e-12
    norms = norm(chroma_sum, axis=0) + eps
    return chroma_sum / norms

def _slice_pianoroll_window(
    pr: np.ndarray,
    fs: float,
    start_s: float = 0.0,
    max_len_s: Optional[float] = None,
) -> np.ndarray:
    """
    Slice a pianoroll-like array (T, D) by time window in seconds.
    Keeps rows in [start_s, start_s + max_len_s).
    """
    if pr.ndim != 2:
        raise ValueError(f"Expected 2D array (T, D); got shape {pr.shape}")

    T = pr.shape[0]
    start_idx = int(round(start_s * fs))
    start_idx = max(0, min(start_idx, T))

    if max_len_s is None:
        end_idx = T
    else:
        end_idx = start_idx + int(round(max_len_s * fs))
        end_idx = max(start_idx, min(end_idx, T))

    return pr[start_idx:end_idx, :]


def chroma_dtw(pm_ref, pm_hyp, fs=100, transpose_invariant:bool=True, include_drums=False,
    start_s: float = 0.0, 
    max_len_s: Optional[float] = None):

    A = _pm_chroma(pm_ref, fs=fs, include_drums=include_drums).T  # (T, 12)
    B = _pm_chroma(pm_hyp, fs=fs, include_drums=include_drums).T  # (T, 12)

    # Apply the same time windowing to both sequences
    A = _slice_pianoroll_window(A, fs=fs, start_s=start_s, max_len_s=max_len_s)
    B = _slice_pianoroll_window(B, fs=fs, start_s=start_s, max_len_s=max_len_s)

    if A.size == 0 or B.size == 0:
        return float("inf")

    def one(X, Y):
        # dtw-python expects the distance callback under the dist_method keyword
        # ("dist" is not a valid argument). The call returns a DTW object whose
        # ``distance`` attribute holds the accumulated cost we need.
        res = dtw(X, Y, dist_method=lambda x, y: norm(x - y))
        return float(res.distance)

    best = one(A, B)
    """
    I dont think the folloing code block is needed for tokenization setting since the tokenizer should not do a pitch shift.
    Nontheless, It could be handy as a metric for evaluating the models after training to see if they learned harmony in some way.
    """
    if transpose_invariant:
        for k in range(1, 12):
            best = min(best, one(A, np.roll(B, k, axis=1)))
    return best


# ---------- main ----------
def midi_roundtrip_metrics_onset_chroma(
    original_mid, reconstructed_mid, onset_tol=0.03, include_drums=False, fs_chroma=100, calculate_transpose_invariant_chroma: bool = True, max_len_s: Optional[float] = None, start_s=0.0 # (optional) start offset
):
    """
    Precision/Recall/F1 on *onset events*; MAE on onset time; MAE on median durations per onset;
    transpose-invariant chroma DTW.
    """
    pm_ref = (
        original_mid
        if isinstance(original_mid, pretty_midi.PrettyMIDI)
        else pretty_midi.PrettyMIDI(original_mid)
    )
    pm_hyp = (
        reconstructed_mid
        if isinstance(reconstructed_mid, pretty_midi.PrettyMIDI)
        else pretty_midi.PrettyMIDI(reconstructed_mid)
    )

    # Onset arrays (deduped)
    ref_on = _collect_onsets(pm_ref, include_drums)
    hyp_on = _collect_onsets(pm_hyp, include_drums)

    if len(ref_on) == 0 and len(hyp_on) == 0:
        # Nothing to compare; return neutral precision/recall/F1 and leave MAEs undefined
        chroma_score = chroma_dtw(
            pm_ref,
            pm_hyp,
            fs=fs_chroma,
            transpose_invariant=calculate_transpose_invariant_chroma,
            include_drums=include_drums,
            start_s=start_s,
            max_len_s=max_len_s,
        )
        chroma_score = float(chroma_score)
        if not np.isfinite(chroma_score):
            chroma_score = float("nan")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "onset_mae_sec": None,
            "dur_mae_sec": None,
            "chroma_dtw": chroma_score,
        }

    # Match onsets within tolerance
    pairs, tp, fp, fn = _match_onsets(ref_on, hyp_on, tol=onset_tol)

    # Precision/Recall/F1

    y_true = np.r_[np.ones(tp), np.ones(fn), np.zeros(fp)]
    y_pred = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp)]
    if y_true.size == 0:
        precision = recall = f1 = 0.0
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

    # onset MAE over matched pairs
    onset_errs = [abs(hyp_on[j] - ref_on[i]) for i, j in pairs]
    onset_mae = float(np.mean(onset_errs)) if onset_errs else None

    # Duration MAE: compare *median* duration of notes at each onset (robust when multiple notes share an onset)
    ref_times, ref_meds = _onset_median_durations(pm_ref, include_drums)
    hyp_times, hyp_meds = _onset_median_durations(pm_hyp, include_drums)

    # Re-match these onset lists with same tolerance to align medians
    i = j = 0
    dur_errs = []
    while i < len(ref_times) and j < len(hyp_times):
        diff = hyp_times[j] - ref_times[i]
        if abs(diff) <= onset_tol:
            dur_errs.append(abs(hyp_meds[j] - ref_meds[i]))
            i += 1
            j += 1
        elif diff < -onset_tol:
            j += 1
        else:
            i += 1
    dur_mae = float(np.mean(dur_errs)) if dur_errs else None

    # Chroma DTW (transpose-invariant)
    chroma_dtw_score = chroma_dtw(
        pm_ref,
        pm_hyp,
        fs=fs_chroma,
        transpose_invariant=True,
        include_drums=include_drums,
    )
    chroma_dtw_score = float(chroma_dtw_score)
    if not np.isfinite(chroma_dtw_score):
        chroma_dtw_score = float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "onset_mae_sec": onset_mae,
        "dur_mae_sec": dur_mae,
        "chroma_dtw": float(chroma_dtw_score),
    }
