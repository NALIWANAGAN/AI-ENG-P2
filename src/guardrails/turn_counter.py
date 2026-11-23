from pathlib import Path
from guardrails.ignored_path import list_unignored_stdlib


def turn_counter(path: Path | str) -> int:
    """Dynamically count the turns according to the repository size."""
    path = Path(path)
    file_count = len(list_unignored_stdlib(path))
    max_turns = max(12, min(60, 10 + file_count // 5))

    return max_turns
