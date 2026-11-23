from pydantic import BaseModel, Field, field_validator

from schema.read_path import FileMetadata


# =============================================
# Request Models
# =============================================


class CreateAuditReportRequest(BaseModel):
    """Input parameters for creating or appending a section to the Markdown audit report."""

    section: str = Field(
        min_length=1,
        max_length=120,
        description="Markdown H2 section heading (e.g. 'Overview', 'Table schema', 'Findings', 'Limitations').",
    )
    content: str = Field(
        min_length=1,
        description="Markdown body content for this section (evidence should cite `path:line`).",
    )
    title: str | None = Field(
        default=None,
        max_length=200,
        description="Top-level `# <Title>` heading used when creating the report on the first call.",
    )
    filename: str | None = Field(
        default=None,
        max_length=120,
        description="Optional `.md` filename inside the allowed output directory (used only on the first call).",
    )
    is_final: bool = Field(
        default=False,
        description="Set to True only on the last call once all report sections are complete.",
    )

    @field_validator("section", "content")
    @classmethod
    def _validate_non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field must be a non-empty string")
        return value


# =============================================
# Response Models
# =============================================


class AuditReportPublicResult(BaseModel):
    """LLM-facing result returned after writing a report section (excludes audit metadata)."""

    file_name: str = Field(description="Relative Markdown filename of the active audit report.")
    section_saved: str = Field(description="Section heading written or appended in this call.")
    sections_written: list[str] = Field(
        default_factory=list,
        description="Ordered list of all section headings currently saved in this report.",
    )
    is_final: bool = Field(
        default=False,
        description="True once the audit report has been finalized and locked.",
    )
    bytes_written: int = Field(
        default=0,
        ge=0,
        description="Current total UTF-8 byte size of the Markdown report file.",
    )


class AuditReportResult(AuditReportPublicResult):
    """Full audit report write result including internal OS metadata."""

    metadata: FileMetadata
