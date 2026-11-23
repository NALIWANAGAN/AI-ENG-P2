import hashlib
import json
import logging
from typing import Any

from schema.guardrail import ToolCallRecord, TurnDecision

logger = logging.getLogger(__name__)


class TurnTracker:
    """Per-run harness guardrail that detects repetitive tool loops and consecutive errors."""

    def __init__(self, max_turns: int = 12, max_repeats: int = 3) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_repeats < 2:
            raise ValueError("max_repeats must be >= 2")

        self._max_turns = max_turns
        self._max_repeats = max_repeats
        self._turn_number = 0
        self._consecutive_errors = 0
        # Compact per-turn fingerprints: (turn_number, call_signature, result_hash)
        self._history: list[tuple[int, tuple[tuple[str, str], ...], str]] = []

    def record_turn(self, tool_calls: list[ToolCallRecord]) -> TurnDecision:
        """Evaluate the current turn's tool calls and return a 'continue', 'warn', or 'stop' decision."""
        self._turn_number += 1
        turn_num = self._turn_number

        signature = self._build_call_signature(tool_calls)
        result_hash = self._hash_results(tool_calls)
        all_errored = bool(tool_calls) and all(tc.is_error for tc in tool_calls)

        if all_errored:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0

        repeated_turns = [
            prev_turn
            for prev_turn, prev_sig, _ in self._history
            if signature and prev_sig == signature
        ]

        self._history.append((turn_num, signature, result_hash))

        is_repetitive = bool(repeated_turns)
        total_occurrences = len(repeated_turns) + 1

        # Hard kill switch: 3x identical calls (original + 2 repeats by default)
        if is_repetitive and total_occurrences >= self._max_repeats:
            msg = (
                f"Stopped: repeated the same tool call(s) {total_occurrences} times "
                f"(turns {repeated_turns + [turn_num]})."
            )
            logger.error("[KILL SWITCH] %s", msg)
            return TurnDecision(
                turn_number=turn_num,
                action="stop",
                is_repetitive=True,
                repeated_turns=repeated_turns,
                system_message=msg,
            )

        # Hard kill switch: consecutive failing turns
        if self._consecutive_errors >= self._max_repeats:
            msg = f"Stopped: {self._consecutive_errors} consecutive turns failed with tool errors."
            logger.error("[KILL SWITCH] %s", msg)
            return TurnDecision(
                turn_number=turn_num,
                action="stop",
                is_repetitive=is_repetitive,
                repeated_turns=repeated_turns,
                system_message=msg,
            )

        # Hard kill switch: max turn budget reached
        if turn_num >= self._max_turns:
            msg = f"Stopped: reached max_turns limit ({self._max_turns})."
            logger.warning("[MAX TURNS] %s", msg)
            return TurnDecision(
                turn_number=turn_num,
                action="stop",
                is_repetitive=is_repetitive,
                repeated_turns=repeated_turns,
                system_message=msg,
            )

        # Actionable warning on first repeat so the model pivots cleanly
        if is_repetitive:
            calls_preview = ", ".join(f"{name}({args})" for name, args in signature)
            msg = (
                f"Repetitive action detected: you already ran {calls_preview} on turn(s) {repeated_turns}. "
                "Do NOT repeat the exact same call. Choose a different action: inspect a different file "
                "or line range, search a different symbol, or conclude your audit with the findings you have."
            )
            logger.warning("[REPETITIVE TURN %d] matches turns %s", turn_num, repeated_turns)
            return TurnDecision(
                turn_number=turn_num,
                action="warn",
                is_repetitive=True,
                repeated_turns=repeated_turns,
                system_message=msg,
            )
            
                
        # warning if 85% of the turn is being reached
        budget_turn = int(0.85 * self._max_turns)
        if self._max_turns >= 4 and turn_num >= budget_turn:
            turns_left = self._max_turns - turn_num
            logger.warning("[SOFT MAX TURN] REACHED 85%% of MAX TURNS: %d", budget_turn)
            msg = (
                f"Budget almost exhausted ({turns_left} turns left). Stop opening new files, "
                "write your remaining findings with create_audit_report, and list unchecked areas under Limitations."
            )
            return TurnDecision(
                turn_number=turn_num,
                action="warn",
                is_repetitive=is_repetitive,
                repeated_turns=repeated_turns,
                system_message=msg,
            )

        # Actionable warning on consecutive tool errors (even with different arguments)
        if self._consecutive_errors >= 2:
            msg = (
                f"Your last {self._consecutive_errors} turns all ended in tool errors. "
                "Check the error messages carefully (e.g., use repo-relative paths from `list_folder_content`) "
                "and try a different valid approach."
            )
            logger.warning("[CONSECUTIVE ERRORS] %d turns in a row", self._consecutive_errors)
            return TurnDecision(
                turn_number=turn_num,
                action="warn",
                is_repetitive=False,
                repeated_turns=[],
                system_message=msg,
            )

        # Normal progress — stay silent
        return TurnDecision(
            turn_number=turn_num,
            action="continue",
            is_repetitive=False,
            repeated_turns=[],
            system_message=None,
        )

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))

    def _build_call_signature(self, tool_calls: list[ToolCallRecord]) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (tc.tool_name.strip().lower(), self._canonical_json(tc.arguments))
                for tc in tool_calls
            )
        )

    def _hash_results(self, tool_calls: list[ToolCallRecord]) -> str:
        canonical = self._canonical_json(
            sorted(
                (tc.tool_name.strip().lower(), self._canonical_json(tc.arguments), self._canonical_json(tc.result))
                for tc in tool_calls
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
