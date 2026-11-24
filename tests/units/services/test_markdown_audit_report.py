from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import pytest
from pydantic import ValidationError

from schema.audit_report import AuditReportResult, CreateAuditReportRequest
from services.markdown_audit_report import MarkdownAuditReport


@pytest.fixture
def report_dirs(tmp_path: Path) -> dict[str, Path]:
    output_dir = tmp_path / "reports"
    sibling_dir = tmp_path / "reports_outside"
    output_dir.mkdir()
    sibling_dir.mkdir()
    (output_dir / "README.md").write_text("# Existing README\n", encoding="utf-8")
    return {"base": tmp_path, "output_dir": output_dir, "sibling_dir": sibling_dir}


class TestMarkdownAuditReportInit:
    def test_happy_valid_output_dir(self, report_dirs: dict[str, Path]) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        assert writer.output_dir == report_dirs["output_dir"].resolve()
        assert writer.active_file is None
        assert writer.is_finalized is False

    def test_sad_invalid_output_dir_and_max_bytes(self, report_dirs: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError):
            MarkdownAuditReport(output_dir=report_dirs["base"] / "missing_dir")
        with pytest.raises(NotADirectoryError):
            MarkdownAuditReport(output_dir=report_dirs["output_dir"] / "README.md")
        with pytest.raises(ValueError):
            MarkdownAuditReport(output_dir=report_dirs["output_dir"], max_report_bytes=0)


class TestMarkdownAuditReportWriteSection:
    def test_happy_incremental_multi_turn_writes_same_file_and_merges_sections(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])

        r1 = writer.write_section(
            CreateAuditReportRequest(
                title="Orders DB Audit",
                filename="orders_audit.md",
                section="Overview",
                content="Audited the orders service (`src/repo.py:1`).",
            )
        )
        assert isinstance(r1, AuditReportResult)
        assert re.fullmatch(r"orders_audit-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", r1.file_name)
        assert r1.section_saved == "Overview"
        assert r1.sections_written == ["Overview"]
        assert r1.is_final is False
        assert r1.bytes_written > 0

        # Turn 2: no filename passed -> automatically writes to the SAME active file
        r2 = writer.write_section(
            CreateAuditReportRequest(
                section="Findings",
                content="- Missing `deleted_at IS NULL` (`src/repo.py:12`).",
            )
        )
        assert r2.file_name == r1.file_name
        assert r2.sections_written == ["Overview", "Findings"]

        # Turn 3 (Option A): repeating 'findings' (different casing) appends under existing `## Findings`
        r3 = writer.write_section(
            CreateAuditReportRequest(
                section="findings",
                content="- Unindexed join column `user_id` (`src/repo.py:45`).",
                is_final=True,
            )
        )
        assert r3.file_name == r1.file_name
        assert r3.section_saved == "Findings"
        assert r3.sections_written == ["Overview", "Findings"]
        assert r3.is_final is True
        assert writer.is_finalized is True

        # Verify clean single-heading Markdown output on disk
        written = (report_dirs["output_dir"] / r1.file_name).read_text(encoding="utf-8")
        assert written.startswith("# Orders DB Audit\n\nDate: ")
        assert written.count("## Findings") == 1
        assert "Missing `deleted_at IS NULL`" in written
        assert "Unindexed join column `user_id`" in written

    def test_happy_auto_generates_unique_md_filename_when_omitted_or_colliding(
        self, report_dirs: dict[str, Path]
    ) -> None:
        w1 = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        w2 = MarkdownAuditReport(output_dir=report_dirs["output_dir"])

        r1 = w1.write_section(CreateAuditReportRequest(section="Overview", content="Run 1"))
        r2 = w2.write_section(CreateAuditReportRequest(section="Overview", content="Run 2"))

        assert re.fullmatch(r"audit_report-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", r1.file_name)
        assert re.fullmatch(r"audit_report-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}(-\d+)?\.md", r2.file_name)
        assert r1.file_name != r2.file_name

        # Same agent-chosen filename across two separate runs never collides or overwrites existing files
        w3 = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        w4 = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        r3 = w3.write_section(CreateAuditReportRequest(filename="README.md", section="Overview", content="Run 3"))
        r4 = w4.write_section(CreateAuditReportRequest(filename="README.md", section="Overview", content="Run 4"))

        assert re.fullmatch(r"README-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", r3.file_name)
        assert re.fullmatch(r"README-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}(-\d+)?\.md", r4.file_name)
        assert r3.file_name != r4.file_name
        assert (report_dirs["output_dir"] / "README.md").read_text(encoding="utf-8") == "# Existing README\n"

    def test_happy_redacts_secrets_and_pii_before_writing_to_disk(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        res = writer.write_section(
            CreateAuditReportRequest(
                title="Audit for postgres://admin:s3cr3tPass@10.0.0.5:5432/app",
                section="Findings",
                content=(
                    "Found hardcoded key sk-proj-1234567890abcdefghijklmnop, "
                    "email alice@example.com, phone +1-415-555-0199, IP 192.168.1.9."
                ),
            )
        )
        on_disk = (report_dirs["output_dir"] / res.file_name).read_text(encoding="utf-8")
        assert "s3cr3tPass" not in on_disk
        assert "sk-proj-1234567890abcdefghijklmnop" not in on_disk
        assert "alice@example.com" not in on_disk
        assert "+1-415-555-0199" not in on_disk
        assert "192.168.1.9" not in on_disk
        assert "[REDACTED_CREDENTIALS]" in on_disk
        assert "[REDACTED_TOKEN]" in on_disk
        assert "[REDACTED_EMAIL]" in on_disk
        assert "[REDACTED_PHONE]" in on_disk
        assert "[REDACTED_IP]" in on_disk

    def test_edge_concurrent_writes_stay_thread_safe(self, report_dirs: dict[str, Path]) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        r0 = writer.write_section(
            CreateAuditReportRequest(filename="parallel.md", section="Overview", content="Start")
        )

        def _append(idx: int) -> None:
            writer.write_section(
                CreateAuditReportRequest(section="Findings", content=f"- Finding #{idx}")
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_append, range(1, 9)))

        text = (report_dirs["output_dir"] / r0.file_name).read_text(encoding="utf-8")
        for idx in range(1, 9):
            assert f"- Finding #{idx}" in text

    def test_sad_write_after_finalization_raises_runtime_error(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        writer.write_section(
            CreateAuditReportRequest(section="Limitations", content="Done.", is_final=True)
        )
        with pytest.raises(RuntimeError, match="already been finalized"):
            writer.write_section(CreateAuditReportRequest(section="Extra", content="Too late"))

    @pytest.mark.parametrize(
        "bad_filename",
        [
            "../outside.md",
            "sub/../../reports_outside/escape.md",
            "../reports_outside/escape.md",
        ],
    )
    def test_sad_path_escape_raises_permission_error(
        self, report_dirs: dict[str, Path], bad_filename: str
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        with pytest.raises(PermissionError, match="outside the allowed output directory"):
            writer.write_section(
                CreateAuditReportRequest(filename=bad_filename, section="Overview", content="x")
            )

    def test_sad_absolute_outside_path_raises_permission_error(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])
        outside_md = str(report_dirs["sibling_dir"] / "escape.md")
        with pytest.raises(PermissionError):
            writer.write_section(
                CreateAuditReportRequest(filename=outside_md, section="Overview", content="x")
            )

    def test_sad_non_markdown_extension_and_null_bytes_rejected(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"])

        with pytest.raises(ValueError, match="\\.md"):
            writer.write_section(
                CreateAuditReportRequest(filename="evil.py", section="Overview", content="x")
            )
        with pytest.raises(ValueError, match="null bytes"):
            writer.write_section(
                CreateAuditReportRequest(filename="report\x00.md", section="Overview", content="x")
            )

    def test_sad_max_report_bytes_exceeded_raises_value_error_without_corrupting_file(
        self, report_dirs: dict[str, Path]
    ) -> None:
        writer = MarkdownAuditReport(output_dir=report_dirs["output_dir"], max_report_bytes=120)
        r1 = writer.write_section(
            CreateAuditReportRequest(filename="small.md", section="Overview", content="Short.")
        )
        before = (report_dirs["output_dir"] / r1.file_name).read_text(encoding="utf-8")

        with pytest.raises(ValueError, match="exceeds max_report_bytes"):
            writer.write_section(CreateAuditReportRequest(section="Findings", content="x" * 200))

        # File on disk remains unchanged after rejected write
        assert (report_dirs["output_dir"] / r1.file_name).read_text(encoding="utf-8") == before

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"section": "", "content": "ok"},
            {"section": "   ", "content": "ok"},
            {"section": "Overview", "content": "   "},
        ],
    )
    def test_sad_blank_request_fields_raise_validation_error(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            CreateAuditReportRequest(**kwargs)
