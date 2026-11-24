from datetime import datetime
from pathlib import Path
import threading

from core.settings import settings
from guardrails.redact import redact_text
from schema.audit_report import AuditReportResult, CreateAuditReportRequest
from schema.read_path import FileMetadata


class MarkdownAuditReport:
    """Sandboxed incremental Markdown report writer with mandatory secret/PII redaction."""

    _DEFAULT_TITLE = "Database Schema Audit Report"

    def __init__(
        self,
        output_dir: Path | str | None = None,
        max_report_bytes: int = settings.MAX_AUDIT_REPORT_BYTES,
    ) -> None:
        if max_report_bytes < 1:
            raise ValueError("max_report_bytes must be >= 1")

        base = Path(output_dir) if output_dir is not None else Path.cwd()
        if not base.exists():
            raise FileNotFoundError(f"Output directory does not exist: {base}")
        if not base.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {base}")

        self._output_dir = base.resolve()
        self._max_report_bytes = max_report_bytes
        self._lock = threading.Lock()

        # Per-run state: keeps all turns writing into the same report file
        self._active_file: Path | None = None
        self._title: str = self._DEFAULT_TITLE
        self._created_date: str | None = None
        self._sections: dict[str, tuple[str, list[str]]] = {}
        self._is_finalized: bool = False

    @property
    def output_dir(self) -> Path:
        """Resolved directory that all report writes are confined to."""
        return self._output_dir

    @property
    def active_file(self) -> Path | None:
        """Path to the currently active Markdown report for this run, if created."""
        return self._active_file

    @property
    def is_finalized(self) -> bool:
        """True once the report has been marked final and locked."""
        return self._is_finalized

    def write_section(self, request: CreateAuditReportRequest) -> AuditReportResult:
        """Create or incrementally append a redacted Markdown section to the active report."""
        with self._lock:
            if self._is_finalized:
                raise RuntimeError(
                    "Audit report has already been finalized and cannot accept further writes."
                )

            clean_section = redact_text(request.section.strip()).lstrip("#").strip()
            if not clean_section:
                raise ValueError("section heading must contain non-heading characters")

            clean_content = redact_text(request.content.strip())
            section_key = clean_section.lower()

            # Resolve or reuse the single active file for this run
            is_first_write = self._active_file is None
            if is_first_write:
                target_file = self._resolve_new_report_path(request.filename)
                title = (
                    redact_text(request.title.strip()).lstrip("#").strip()
                    if request.title and request.title.strip()
                    else self._DEFAULT_TITLE
                )
                created_date = datetime.now().strftime("%Y-%m-%d")
            else:
                assert self._active_file is not None
                target_file = self._active_file
                title = self._title
                created_date = self._created_date or datetime.now().strftime("%Y-%m-%d")

            # Prepare updated sections (Option A: append under existing heading if repeated)
            next_sections = {
                key: (heading, list(blocks)) for key, (heading, blocks) in self._sections.items()
            }
            if section_key in next_sections:
                existing_heading, blocks = next_sections[section_key]
                blocks.append(clean_content)
                saved_heading = existing_heading
            else:
                next_sections[section_key] = (clean_section, [clean_content])
                saved_heading = clean_section

            rendered = self._render_markdown(title, created_date, next_sections)
            encoded = rendered.encode("utf-8")
            if len(encoded) > self._max_report_bytes:
                raise ValueError(
                    f"Audit report size ({len(encoded)} bytes) exceeds max_report_bytes ({self._max_report_bytes})."
                )

            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_bytes(encoded)

            # Commit state only after successful disk write
            self._active_file = target_file
            self._title = title
            self._created_date = created_date
            self._sections = next_sections
            if request.is_final:
                self._is_finalized = True

            return AuditReportResult(
                file_name=target_file.relative_to(self._output_dir).as_posix(),
                section_saved=saved_heading,
                sections_written=[heading for heading, _ in self._sections.values()],
                is_final=self._is_finalized,
                bytes_written=len(encoded),
                metadata=self._file_metadata(target_file),
            )

    def _resolve_new_report_path(self, filename: str | None) -> Path:
        """Resolve a unique timestamped `.md` path (`<name>-mm-dd-yyyy-HH-MM.md`) strictly inside `self._output_dir`."""
        stamp = datetime.now().strftime("%m-%d-%Y-%H-%M")

        if filename is not None:
            if "\x00" in filename:
                raise ValueError("Filename must not contain null bytes")
            stripped = filename.strip()
            if not stripped:
                raise ValueError("Filename must not be blank")

            candidate = Path(stripped)
            if not candidate.is_absolute():
                candidate = self._output_dir / candidate

            resolved = candidate.resolve()
            if not resolved.is_relative_to(self._output_dir):
                raise PermissionError(
                    f"Access denied: report path '{filename}' resolves outside the allowed output directory."
                )
            if resolved.suffix.lower() != ".md":
                raise ValueError("Audit report filename must end with '.md'")

            parent = resolved.parent
            stem = resolved.stem
        else:
            parent = self._output_dir
            stem = "audit_report"

        final_path = (parent / f"{stem}-{stamp}.md").resolve()
        counter = 1
        while final_path.exists():
            final_path = (parent / f"{stem}-{stamp}-{counter}.md").resolve()
            counter += 1

        if not final_path.is_relative_to(self._output_dir):
            raise PermissionError(
                f"Access denied: report path '{filename}' resolves outside the allowed output directory."
            )
        return final_path

    @staticmethod
    def _render_markdown(
        title: str,
        created_date: str,
        sections: dict[str, tuple[str, list[str]]],
    ) -> str:
        parts = [f"# {title}\n\nDate: {created_date} | Writer: Agent"]
        for heading, blocks in sections.values():
            body = "\n\n".join(blocks)
            parts.append(f"## {heading}\n\n{body}")
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _file_metadata(path: Path) -> FileMetadata:
        resolved = path.resolve()
        stat = resolved.stat()
        birthtime = getattr(stat, "st_birthtime", None)
        return FileMetadata(
            file_name=resolved.name,
            file_path=str(resolved),
            creation_date=datetime.fromtimestamp(birthtime) if birthtime is not None else None,
            modification_date=datetime.fromtimestamp(stat.st_mtime),
        )
