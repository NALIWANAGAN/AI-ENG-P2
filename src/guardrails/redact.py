import re

_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV6_PATTERN = (
    r"(?<![0-9a-fA-F:])(?:"
    r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
    r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){0,5}"
    r")(?![0-9a-fA-F:])"
)

_REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    # Private key blocks
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # DB / service connection strings with embedded credentials (preserves dialect & host/db)
    (
        re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s:@]+:[^/\s@]+)@"),
        r"\1[REDACTED_CREDENTIALS]@",
    ),
    # Known provider API keys & tokens (OpenAI, Anthropic, Gemini, AWS, GitHub, Slack, JWT, Bearer)
    (
        re.compile(
            r"\b("
            r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
            r"|AIza[0-9A-Za-z_-]{35}"
            r"|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9_]{36,}"
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"
            r"|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            r")\b"
        ),
        "[REDACTED_TOKEN]",
    ),
    (
        re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{16,}"),
        r"\1[REDACTED_TOKEN]",
    ),
    # Hardcoded secret assignments (preserves variable/column name, masks literal value)
    (
        re.compile(
            r"(?i)(\b[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|PASSWD|TOKEN|PRIVATE_KEY)[A-Z0-9_]*\s*[:=]\s*['\"])([^'\"]{6,})(['\"])"
        ),
        r"\1[REDACTED_SECRET]\3",
    ),
    # User PII: email addresses
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    # User PII: US SSNs (matched before phone numbers)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    # IP addresses (IPv4 and IPv6)
    (
        re.compile(rf"(?<!\d\.)(?<!\d){_IPV4_OCTET}(?:\.{_IPV4_OCTET}){{3}}(?!\.\d)(?!\d)"),
        "[REDACTED_IP]",
    ),
    (
        re.compile(_IPV6_PATTERN),
        "[REDACTED_IP]",
    ),
    # User PII: phone numbers (international + formatted local, avoiding raw timestamps/IDs)
    (
        re.compile(
            r"(?<!\w)\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"
            r"|(?<!\d)(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"
        ),
        "[REDACTED_PHONE]",
    ),
]


def redact_text(text: str) -> str:
    """Mask secrets, credentials, IPs, and user PII while preserving code and SQL structure."""
    if not text:
        return text

    redacted = text
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted
