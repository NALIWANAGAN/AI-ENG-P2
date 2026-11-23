import pytest

from guardrails.redact import redact_text


class TestRedactText:
    def test_happy_redacts_private_key_block(self) -> None:
        raw = (
            "cert = '''\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA1234567890abcdef\n"
            "-----END RSA PRIVATE KEY-----\n"
            "'''"
        )
        result = redact_text(raw)
        assert "[REDACTED_PRIVATE_KEY]" in result
        assert "MIIEpAIBAAKCAQEA" not in result

    def test_happy_redacts_db_connection_string_preserving_dialect_and_db(self) -> None:
        raw = "DATABASE_URL = 'postgres://admin:s3cr3tPass@db.internal:5432/orders_db'"
        result = redact_text(raw)
        assert result == "DATABASE_URL = 'postgres://[REDACTED_CREDENTIALS]@db.internal:5432/orders_db'"

    def test_happy_redacts_db_connection_string_with_ip_host(self) -> None:
        raw = "postgres://admin:s3cr3tPass@10.0.4.12:5432/orders_db"
        result = redact_text(raw)
        assert result == "postgres://[REDACTED_CREDENTIALS]@[REDACTED_IP]:5432/orders_db"

    @pytest.mark.parametrize(
        "token",
        [
            "sk-proj-1234567890abcdefghijklmnop",
            "AIzaSyD1234567890abcdefghijklmnopqrstuv",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz12",
            "xoxb-1234567890-abcdef1234",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        ],
    )
    def test_happy_redacts_provider_tokens(self, token: str) -> None:
        result = redact_text(f"val = '{token}'")
        assert "[REDACTED_TOKEN]" in result
        assert token not in result

    def test_happy_redacts_bearer_header(self) -> None:
        raw = "Authorization: Bearer super_secret_bearer_token_12345"
        assert redact_text(raw) == "Authorization: Bearer [REDACTED_TOKEN]"

    def test_happy_redacts_hardcoded_secret_assignment_preserving_key_name(self) -> None:
        raw = 'OLLAMA_API_KEY = "my-hardcoded-secret-value"\npassword: "hunter22"'
        result = redact_text(raw)
        assert 'OLLAMA_API_KEY = "[REDACTED_SECRET]"' in result
        assert 'password: "[REDACTED_SECRET]"' in result
        assert "my-hardcoded-secret-value" not in result
        assert "hunter22" not in result

    def test_happy_redacts_emails_and_ssns(self) -> None:
        raw = "INSERT INTO users (email, ssn) VALUES ('alice.doe@example.com', '123-45-6789');"
        result = redact_text(raw)
        assert result == "INSERT INTO users (email, ssn) VALUES ('[REDACTED_EMAIL]', '[REDACTED_SSN]');"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("host = '192.168.1.42'", "host = '[REDACTED_IP]'"),
            ("bind 10.0.0.1:8080", "bind [REDACTED_IP]:8080"),
            ("ip = '2001:0db8:85a3:0000:0000:8a2e:0370:7334'", "ip = '[REDACTED_IP]'"),
            ("ip = '2001:db8::1'", "ip = '[REDACTED_IP]'"),
            ("ip = 'fe80::1ff:fe23:4567:890a'", "ip = '[REDACTED_IP]'"),
        ],
    )
    def test_happy_redacts_ipv4_and_ipv6_addresses(self, raw: str, expected: str) -> None:
        assert redact_text(raw) == expected

    @pytest.mark.parametrize(
        "phone",
        [
            "+1-415-555-0199",
            "+1 (415) 555-0199",
            "+63 917 123 4567",
            "(415) 555-0199",
            "415-555-0199",
            "415.555.0199",
        ],
    )
    def test_happy_redacts_phone_numbers(self, phone: str) -> None:
        result = redact_text(f"phone = '{phone}'")
        assert result == "phone = '[REDACTED_PHONE]'"

    def test_edge_preserves_sql_schema_code_dates_timestamps_and_invalid_ips(self) -> None:
        safe_code = "\n".join(
            [
                "CREATE TABLE users (",
                "    id SERIAL PRIMARY KEY,",
                "    email VARCHAR(255) NOT NULL,",
                "    password_hash TEXT NOT NULL,",
                "    phone_number VARCHAR(20),",
                "    created_at TIMESTAMP DEFAULT '2026-11-23 12:30:45'",
                ");",
                "unix_ts = 1732320000",
                "bad_ip = '999.999.999.999'",
                "cpp_ref = 'std::vector::size'",
            ]
        )
        assert redact_text(safe_code) == safe_code

    def test_edge_empty_string_returns_empty_string(self) -> None:
        assert redact_text("") == ""
