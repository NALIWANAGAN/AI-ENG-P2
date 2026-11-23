from pathlib import Path

import pytest

from guardrails.turn_counter import turn_counter


class TestTurnCounter:
    def test_happy_small_repo_uses_minimum_floor_of_12(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"file_{i}.py").write_text("x = 1\n", encoding="utf-8")

        assert turn_counter(tmp_path) == 12
        assert turn_counter(str(tmp_path)) == 12

    def test_happy_medium_repo_scales_with_unignored_file_count(self, tmp_path: Path) -> None:
        (tmp_path / ".gitignore").write_text("*.log\n", encoding="utf-8")
        # 50 unignored files (+ .gitignore = 51 -> 10 + 51 // 5 = 20)
        for i in range(50):
            (tmp_path / f"mod_{i}.py").write_text("pass\n", encoding="utf-8")
        # Ignored files must not inflate turn budget
        for i in range(100):
            (tmp_path / f"noise_{i}.log").write_text("ignored\n", encoding="utf-8")

        assert turn_counter(tmp_path) == 20

    def test_happy_large_repo_caps_at_60(self, tmp_path: Path) -> None:
        for i in range(300):
            (tmp_path / f"f_{i}.py").write_text("pass\n", encoding="utf-8")

        assert turn_counter(tmp_path) == 60

    def test_edge_empty_directory_returns_12(self, tmp_path: Path) -> None:
        assert turn_counter(tmp_path) == 12

    def test_sad_missing_or_file_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            turn_counter(tmp_path / "does_not_exist")

        single_file = tmp_path / "only.py"
        single_file.write_text("pass\n", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            turn_counter(single_file)
