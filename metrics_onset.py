# pip install pretty_midi numpy dtw-python scikit-learn

import numpy as np
import pretty_midi
from dtw import dtw
from numpy.linalg import norm

try:
    from sklearn.metrics import precision_recall_fscore_support

    _HAS_SK = True
except Exception:
    _HAS_SK = False


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
    from collections import defaultdict

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
    """
    Sum instrument chroma across (non-drum) instruments using get_chroma.
    """
    chroma_sum = None
    for inst in pm.instruments:
        if (not include_drums) and inst.is_drum:
            continue
        C = inst.get_chroma(fs=fs)  # (12, T)
        chroma_sum = C if chroma_sum is None else chroma_sum + C
    if chroma_sum is None:
        return np.zeros((12, 0), dtype=np.float32)
    # L2-normalize each frame to reduce loudness bias
    eps = 1e-12
    norms = norm(chroma_sum, axis=0) + eps
    return chroma_sum / norms


def chroma_dtw(pm_ref, pm_hyp, fs=100, transpose_invariant=True, include_drums=False):
    A = _pm_chroma(pm_ref, fs=fs, include_drums=include_drums).T  # (T, 12)
    B = _pm_chroma(pm_hyp, fs=fs, include_drums=include_drums).T  # (T, 12)
    if A.size == 0 or B.size == 0:
        return float("inf")

    def one(X, Y):
        d, _, _, _ = dtw(X, Y, dist=lambda x, y: norm(x - y))
        return float(d)

    best = one(A, B)
    """
    I dont think the folloing code block is needed for tokenization setting since the tokenizer should not do a pitch shift.
    Nontheless, I could be handy as a metric for evaluating the models after training to see if they learned harmony in some way.
    """
    if transpose_invariant: 
        for k in range(1, 12):
            best = min(best, one(A, np.roll(B, k, axis=1)))
    return best


# ---------- main ----------
def midi_roundtrip_metrics_onset_chroma(
    original_mid, reconstructed_mid, onset_tol=0.03, include_drums=False, fs_chroma=100
):
    """
    Precision/Recall/F1 on *onset events*; MAE on onset time; MAE on median durations per onset;
    transpose-invariant chroma DTW.
    """
    pm_ref = pretty_midi.PrettyMIDI(original_mid)
    pm_hyp = pretty_midi.PrettyMIDI(reconstructed_mid)

    # Onset arrays (deduped)
    ref_on = _collect_onsets(pm_ref, include_drums)
    hyp_on = _collect_onsets(pm_hyp, include_drums)

    # Match onsets within tolerance
    pairs, tp, fp, fn = _match_onsets(ref_on, hyp_on, tol=onset_tol)

    # Precision/Recall/F1

    y_true = np.r_[np.ones(tp), np.ones(fn), np.zeros(fp)]
    y_pred = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp)]
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

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "onset_mae_sec": onset_mae,
        "dur_mae_sec": dur_mae,
        "chroma_dtw": float(chroma_dtw_score),
    }
