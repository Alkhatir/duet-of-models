#!/usr/bin/env python3
"""
MidiTok YAML configuration builder.

- Covers every TokenizerConfig option (per docs).
- Adds tokenizer-specific extras:
  * REMI: max_bar_embedding, use_bar_end_tokens, add_trailing_bars
  * MIDILike: max_duration
  * MMM: density_bins_max
  * PerTok: ticks_per_quarter, use_microtiming, max_microtiming_shift,
           num_microtiming_bins, use_position_toks
- Validates extras vs tokenizer compatibility.
- Exports/loads YAML.

Docs sources:
- TokenizerConfig parameters
- Tokenizer pages (REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM, PerTok)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Sequence, Mapping, Optional, List
import copy
import argparse
from utils import iter_midi_paths
import yaml
from miditok import MusicTokenizer


# ------------------------------
# Supported tokenizers & extras
# ------------------------------
SUPPORTED_TOKENIZERS = {
    "REMI",
    "MIDILike",
    "TSD",
    "Structured",
    "CPWord",
    "Octuple",
    "MuMIDI",
    "MMM",
    "PerTok",
}

# Extras accepted per tokenizer (with defaults).
TOKENIZER_EXTRAS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "REMI": {
        "max_bar_embedding": None,  # int or None
        "use_bar_end_tokens": False,  # bool
        "add_trailing_bars": False,  # bool
    },
    "MIDILike": {
        # tuple of three ints: (num_beats, num_frames, res_frames)
        # Put it under additional_params["max_duration"] in MidiTok
        "max_duration": None,
    },
    "MMM": {
        # tuple: (num_bins, max_density_notes_per_beat)
        "density_bins_max": (10, 20),
    },
    "PerTok": {
        "ticks_per_quarter": 480,  # int
        "use_microtiming": False,  # bool
        "max_microtiming_shift": 0.125,  # float (beats)
        "num_microtiming_bins": 30,  # int
        "use_position_toks": True,  # bool (PerTok lists this in example)
    },
    # Tokenizers with no extra keys beyond TokenizerConfig:
    "TSD": {},
    "Structured": {},
    "CPWord": {},
    "Octuple": {},
    "MuMIDI": {},
}


# ------------------------------
# Full TokenizerConfig defaults (from docs)
# ------------------------------
def _range_list(start: int, end: int) -> List[int]:
    return list(range(start, end + 1))


@dataclass
class TokenizerConfigFields:
    # Core ranges / resolutions
    pitch_range: Tuple[int, int] = (21, 109)
    beat_res: Dict[Tuple[int, int], int] = field(
        default_factory=lambda: {(0, 4): 8, (4, 12): 4}
    )
    num_velocities: int = 32
    special_tokens: Sequence[str] = field(
        default_factory=lambda: ["PAD", "BOS", "EOS", "MASK"]
    )
    encode_ids_split: str = "bar"  # "bar" | "beat" | "no"

    # Toggles for token types
    use_velocities: bool = True
    use_note_duration_programs: Sequence[int] = field(
        default_factory=lambda: _range_list(-1, 127)
    )
    use_chords: bool = False
    use_rests: bool = False
    use_tempos: bool = False
    use_time_signatures: bool = False
    use_sustain_pedals: bool = False
    use_pitch_bends: bool = False
    use_programs: bool = False
    use_pitch_intervals: bool = False
    use_pitchdrum_tokens: bool = True

    # Durations / rests
    default_note_duration: float = 0.5
    beat_res_rest: Dict[Tuple[int, int], int] = field(
        default_factory=lambda: {(0, 1): 8, (1, 2): 4, (2, 12): 2}
    )

    # Chords
    chord_maps: Dict[str, tuple] = field(
        default_factory=lambda: {
            "maj": (0, 4, 7),
            "min": (0, 3, 7),
            "aug": (0, 4, 8),
            "dim": (0, 3, 6),
            "7maj": (0, 4, 7, 11),
            "7min": (0, 3, 7, 10),
            "7dom": (0, 4, 7, 10),
            "7halfdim": (0, 3, 6, 10),
            "7aug": (0, 4, 8, 11),
            "7dim": (0, 3, 6, 9),
            "9maj": (0, 4, 7, 10, 14),
            "9min": (0, 4, 7, 10, 13),
            "sus2": (0, 2, 7),
            "sus4": (0, 5, 7),
        }
    )
    chord_tokens_with_root_note: bool = False
    chord_unknown: Optional[Tuple[int, int]] = None

    # Tempos
    num_tempos: int = 32
    tempo_range: Tuple[int, int] = (40, 250)
    log_tempos: bool = False
    delete_equal_successive_tempo_changes: bool = False

    # Cleaning / dedup
    remove_duplicated_notes: bool = False

    # Time signatures
    time_signature_range: Mapping[int, Sequence[int] | Tuple[int, int]] = field(
        default_factory=lambda: {8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]}
    )
    sustain_pedal_duration: bool = False
    pitch_bend_range: Tuple[int, int, int] = (-8192, 8191, 32)
    delete_equal_successive_time_sig_changes: bool = False

    # Programs
    programs: Sequence[int] = field(default_factory=lambda: _range_list(-1, 127))
    one_token_stream_for_programs: bool = True
    program_changes: bool = False

    # Pitch intervals
    max_pitch_interval: int = 16
    pitch_intervals_max_time_dist: float = 1.0

    # Drums
    drums_pitch_range: Tuple[int, int] = (27, 88)

    # Attribute controls (all off by default per docs)
    ac_polyphony_track: bool = False
    ac_polyphony_bar: bool = False
    ac_polyphony_min: int = 1
    ac_polyphony_max: int = 6
    ac_pitch_class_bar: bool = False
    ac_note_density_track: bool = False
    ac_note_density_track_min: int = 0
    ac_note_density_track_max: int = 18
    ac_note_density_bar: bool = False
    ac_note_density_bar_max: int = 18
    ac_note_duration_bar: bool = False
    ac_note_duration_track: bool = False
    ac_repetition_track: bool = False
    ac_repetition_track_num_bins: int = 10
    ac_repetition_track_num_consec_bars: int = 4

    # **Anything else** one want to persist can go in additional_params (below)
    # and will be passed through / saved to config.additional_params in MidiTok.


@dataclass
class MidiTokBuilder:
    tokenizer: str = "REMI"
    config: TokenizerConfigFields = field(default_factory=TokenizerConfigFields)
    # Extra per-tokenizer options (e.g., REMI, MIDILike, MMM, PerTok) and any user custom entries
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.tokenizer = self._normalize_tokenizer(self.tokenizer)
        if self.tokenizer not in SUPPORTED_TOKENIZERS:
            raise ValueError(
                f"Unsupported tokenizer '{self.tokenizer}'. "
                f"Choose one of: {sorted(SUPPORTED_TOKENIZERS)}"
            )
        # Preload defaults for tokenizer-specific extras if not provided
        defaults = TOKENIZER_EXTRAS_DEFAULTS.get(self.tokenizer, {})
        for k, v in defaults.items():
            self.additional_params.setdefault(k, copy.deepcopy(v))

    # ---------- Public API ----------
    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dict like:
        {
          'tokenizer': 'REMI',
          'tokenizer_config': {...},   # all TokenizerConfig fields
          'additional_params': {...}   # tokenizer-specific + user extras
        }
        """
        self._validate()
        return {
            "tokenizer": self.tokenizer,
            "tokenizer_config": asdict(self.config),
            "additional_params": copy.deepcopy(self.additional_params),
        }

    def to_yaml(self, path: Optional[str] = None):
        """
        Dump YAML string. If path is given, also write the file.
        """
        data = self.to_dict()
        if path:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, stream=f, sort_keys=False, allow_unicode=True, encoding="utf-8")
            return
        
        

    @classmethod
    def from_yaml(cls, yaml_text_or_path: str) -> "MidiTokBuilder":
        """
        Load from YAML string or file path.
        """
        try:
            # try path
            with open(yaml_text_or_path, "r", encoding="utf-8") as f:
                data = yaml.unsafe_load(f)
        except FileNotFoundError:
            data = yaml.safe_load(yaml_text_or_path)

        tokenizer = data.get("tokenizer", "REMI")
        cfg_dict = data.get("tokenizer_config", {})
        extras = data.get("additional_params", {})
        cfg = TokenizerConfigFields(**cfg_dict)
        return cls(tokenizer=tokenizer, config=cfg, additional_params=extras)

    # ---------- Convenience presets ----------
    @classmethod
    def preset_remi_plus(cls) -> "MidiTokBuilder":
        """
        REMI+ == REMI with programs (single stream) + time signatures.
        """
        cfg = TokenizerConfigFields(
            use_programs=True,
            one_token_stream_for_programs=True,
            use_time_signatures=True,
        )
        return cls(tokenizer="REMI", config=cfg)

    @classmethod
    def preset_octuple_multitrack(cls) -> "MidiTokBuilder":
        """
        Octuple with common multitrack toggles.
        """
        cfg = TokenizerConfigFields(
            use_programs=True,  # Octuple uses Program natively; safe to keep True
            use_tempos=True,
            use_time_signatures=False,  # often used/tested without per docs note
        )
        return cls(tokenizer="Octuple", config=cfg)

    # ---------- Validation & helpers ----------
    def _validate(self):
        """
        - Check extras belong to the selected tokenizer.
        - Spot obvious conflicts and provide helpful errors.
        """
        extras_allowed = set(TOKENIZER_EXTRAS_DEFAULTS.get(self.tokenizer, {}).keys())
        for key in self.additional_params.keys():
            # Always allow custom user keys, but warn on common mistakes
            if (
                key in {"max_bar_embedding", "use_bar_end_tokens", "add_trailing_bars"}
                and self.tokenizer != "REMI"
            ):
                raise ValueError(
                    f"'{key}' is specific to REMI; current tokenizer is {self.tokenizer}."
                )
            if key == "max_duration" and self.tokenizer != "MIDILike":
                raise ValueError(
                    f"'max_duration' is specific to MIDILike; current tokenizer is {self.tokenizer}."
                )
            if key == "density_bins_max" and self.tokenizer != "MMM":
                raise ValueError(
                    f"'density_bins_max' is specific to MMM; current tokenizer is {self.tokenizer}."
                )
            if (
                key
                in {
                    "ticks_per_quarter",
                    "use_microtiming",
                    "max_microtiming_shift",
                    "num_microtiming_bins",
                    "use_position_toks",
                }
                and self.tokenizer != "PerTok"
            ):
                raise ValueError(
                    f"'{key}' is specific to PerTok; current tokenizer is {self.tokenizer}."
                )

        # Program-usage sanity checks
        if self.config.use_rests and self.config.use_programs:
            # Docs note: when using rests with programs, Duration programs list
            # should align with programs; we just warn via error if obviously empty.
            if not self.config.use_note_duration_programs:
                raise ValueError(
                    "When using rests with programs, 'use_note_duration_programs' should include the "
                    "programs you intend to use (docs recommendation)."
                )

        # Per-tokenizer quirks
        if self.tokenizer == "Octuple" and self.config.use_time_signatures:
            # Docs mention Octuple is "implemented with Time Signature but tested without"
            # Not an error, but a caution—could be logged; we keep it permissive.
            pass

        # Ensure encode_ids_split has valid value
        if self.config.encode_ids_split not in {"bar", "beat", "no"}:
            raise ValueError("encode_ids_split must be one of {'bar','beat','no'}.")

    @staticmethod
    def _normalize_tokenizer(name: str) -> str:
        return name.strip().replace("+", "Plus") if name else "REMI"

    def to_MidiTok(self) -> MusicTokenizer:
        """
        Create a miditok.MidiTok instance with this config.
        """
        import miditok

        self._validate()
        cfg_dict = asdict(self.config)
        # Remove any None values from cfg_dict to use miditok defaults
        cfg_dict = {k: v for k, v in cfg_dict.items() if v is not None}
        tokenizer_cls = getattr(miditok, self.tokenizer)
        return tokenizer_cls(
            tokenizer_config=miditok.TokenizerConfig(**(cfg_dict | self.additional_params))
        )


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="MidiTok tokenizer and config YAML builder"
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=True,
        type=str,
        help="Input folder containing MIDI files",
    )
    ap.add_argument(
        "--out",
        dest="out",
        required=True,
        type=str,
        help="Output folder for tokenized MIDIs",
    )
    ap.add_argument(
        "--load-config",
        dest="config",
        required=False,
        type=str,
        help="YAML config file to load containing the tokenization config",
    )
    ap.add_argument(
        "--write-manifest",
        dest="manifest",
        required=False,
        action="store_true",
        help="If set, write a manifest.json file in the output folder",
    )
    ap.add_argument(
        "--limit",
        dest="limit",
        required=False,
        type=int,
        default=None,
        help="If set, limit the number of MIDI files to process (for testing)",
    )
    args = ap.parse_args()

    builder = None
    if not args.config:
        # 1) A generic REMI config with some popular toggles
        builder = MidiTokBuilder(
            tokenizer="REMI",
            additional_params={
                "max_bar_embedding": 512,
                "use_bar_end_tokens": False,
                "add_trailing_bars": False,
            },
        )
        # Toggle useful extras
        builder.config.use_programs = True
        builder.config.one_token_stream_for_programs = True  # REMI+ style
        builder.config.use_time_signatures = True
        builder.config.use_tempos = True
        builder.config.num_tempos = 48
        builder.config.tempo_range = (30, 240)
    else:
        builder = MidiTokBuilder.from_yaml(args.config)

    tokenizer = builder.to_MidiTok()
    midi_paths = list(iter_midi_paths(args.inp))
    if args.limit:
        midi_paths = midi_paths[: args.limit]
    if args.manifest:
        yaml.dump(
            midi_paths, encoding="utf-8", stream=open(f"{args.out}/manifest.yaml", "w")
        )
    print(
        f"Tokenizing {len(midi_paths)} MIDI files from '{args.inp}' to '{args.out}' using {builder.tokenizer}..."
    )
    tokenizer.tokenize_dataset(midi_paths, out_dir=args.out)
