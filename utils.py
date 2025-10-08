from pathlib import Path
from typing import Iterable


def iter_midi_paths(root: Path) -> Iterable[Path]:
    """Yield all .mid/.midi files under a root directory recursively."""
    for ext in (".mid", ".midi"):
        yield from root.rglob(f"*{ext}")
