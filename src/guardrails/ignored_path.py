import fnmatch
from pathlib import Path


def list_unignored_stdlib(path: Path, root_dir: Path | None = None) -> list[Path]:
    path = path.resolve()
    root = root_dir.resolve() if root_dir is not None else path

    patterns: list[str] = [".git/"]
    ignore_files = [root / ".gitignore"]
    if path != root:
        ignore_files.append(path / ".gitignore")

    for ignore_file in ignore_files:
        if ignore_file.is_file():
            for raw_line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)

    def _is_ignored(rel_posix: str, name: str, is_dir: bool) -> bool:
        ignored = False
        for pat in patterns:
            negated = pat.startswith("!")
            raw_pat = pat[1:] if negated else pat
            if not raw_pat:
                continue

            dir_only = raw_pat.endswith("/")
            if dir_only and not is_dir:
                continue

            trimmed = raw_pat.rstrip("/")
            anchored = "/" in trimmed
            clean_pat = trimmed.lstrip("/")

            matched = (
                fnmatch.fnmatch(rel_posix, clean_pat)
                if anchored
                else fnmatch.fnmatch(name, clean_pat) or fnmatch.fnmatch(rel_posix, clean_pat)
            )
            if matched:
                ignored = not negated
        return ignored

    results: list[Path] = []
    visited_dirs: set[Path] = {path}

    def _walk(current_dir: Path) -> None:
        for entry in current_dir.iterdir():
            resolved_entry = entry.resolve()
            # Block symlinks/junctions that escape outside root_dir
            if not resolved_entry.is_relative_to(root):
                continue

            rel_posix = entry.relative_to(root).as_posix()
            is_dir = resolved_entry.is_dir()
            if _is_ignored(rel_posix, entry.name, is_dir):
                continue
            if is_dir:
                if resolved_entry in visited_dirs:
                    continue
                visited_dirs.add(resolved_entry)
                _walk(resolved_entry)
            elif resolved_entry.is_file():
                results.append(resolved_entry)

    _walk(path)
    return results
