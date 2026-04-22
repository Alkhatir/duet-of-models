from src.evaluation.music_metrics import midi_roundtrip_metrics_onset_chroma

resutls = midi_roundtrip_metrics_onset_chroma(
    original_mid="data/lmd_matched/A/A/A/TRAAAGR128F425B14B/1d9d16a9da90c090809c153754823c2b.mid",
    reconstructed_mid="data/lmd_matched/A/A/A/TRAAAGR128F425B14B/dac3cdd0db6341d8dc14641e44ed0d44.mid",
    onset_tol=0.03,
    include_drums=False,
    fs_chroma=2,
    calculate_transpose_invariant_chroma=False,
    max_len_s=None
)

print(resutls)
