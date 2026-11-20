import os
from pathlib import Path

import pytest

from guardrails.ignored_path import list_unignored_stdlib


@pytest.fixture
def gitignore_repo(tmp_path: Path) -> dict[str, Path]:
    """Isolated repo with `.gitignore` rules (wildcards, negation, dir-only, root-anchored)."""
    project = tmp_path / "project"
    outside = tmp_path / "project_secret"
    project.mkdir()
    outside.mkdir()
    (outside / "outside.txt").write_text("outside secret\n", encoding="utf-8")

    (project / ".gitignore").write_text(
        "\n".join(
            [
                "# Comments should be ignored",
                "*.log",
                "!keep.log",
                "build/",
                "/root_only.txt",
                "ignored_dir/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    (project / "ignored_dir").mkdir()
    (project / "ignored_dir" / "hidden.py").write_text("total_cents = 999\n", encoding="utf-8")
    (project / "debug.log").write_text("total_cents debug\n", encoding="utf-8")
    (project / "keep.log").write_text("kept log line\n", encoding="utf-8")
    (project / "build").write_text("regular file named build\n", encoding="utf-8")
    (project / "root_only.txt").write_text("ignored at root\n", encoding="utf-8")

    src = project / "src"
    src.mkdir()
    (src / "root_only.txt").write_text("kept inside src\n", encoding="utf-8")
    (src / "repo.py").write_text("class OrderRepo: pass\n", encoding="utf-8")
    (src / "schema.sql").write_text("CREATE TABLE orders (id INT);\n", encoding="utf-8")
    (src / "build").mkdir()
    (src / "build" / "generated.py").write_text("total_cents = 0\n", encoding="utf-8")

    return {"project": project, "outside": outside}


def test_happy_lists_unignored_files_respecting_gitignore(gitignore_repo: dict[str, Path]) -> None:
    root = gitignore_repo["project"]
    listed = sorted(
        entry.relative_to(root).as_posix()
        for entry in list_unignored_stdlib(root, root)
    )

    # Included files
    assert "build" in listed
    assert "keep.log" in listed
    assert "src/root_only.txt" in listed
    assert "src/schema.sql" in listed
    assert "src/repo.py" in listed

    # Excluded by .gitignore or default .git/
    assert ".git/HEAD" not in listed
    assert "debug.log" not in listed
    assert "root_only.txt" not in listed
    assert "ignored_dir/hidden.py" not in listed
    assert "src/build/generated.py" not in listed


def test_happy_subdirectory_applies_root_gitignore(gitignore_repo: dict[str, Path]) -> None:
    root = gitignore_repo["project"]
    listed = sorted(
        entry.relative_to(root).as_posix()
        for entry in list_unignored_stdlib(root / "src", root)
    )
    assert listed == ["src/repo.py", "src/root_only.txt", "src/schema.sql"]


def test_edge_escaping_symlink_is_skipped(gitignore_repo: dict[str, Path]) -> None:
    root = gitignore_repo["project"]
    link_path = root / "escape_link.txt"
    try:
        os.symlink(gitignore_repo["outside"] / "outside.txt", link_path)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlink creation")

    listed = [
        entry.relative_to(root).as_posix()
        for entry in list_unignored_stdlib(root, root)
    ]
    assert "escape_link.txt" not in listed
