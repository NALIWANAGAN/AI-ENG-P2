import pytest

from guardrails.track_turns import TurnTracker
from schema.guardrail import ToolCallRecord


class TestTurnTracker:
    def test_happy_distinct_turns_and_pagination_continue_silently(self) -> None:
        tracker = TurnTracker(max_turns=10, max_repeats=3)

        d1 = tracker.record_turn(
            [ToolCallRecord(tool_name="list_folder_content", arguments={"path": "."}, result={"content": "a.py"})]
        )
        d2 = tracker.record_turn(
            [ToolCallRecord(tool_name="read_file_slice", arguments={"path": "a.py", "line_start": 1}, result={"content": "x"})]
        )
        # Same tool with different line_start (valid pagination) must NOT trigger warning
        d3 = tracker.record_turn(
            [ToolCallRecord(tool_name="read_file_slice", arguments={"path": "a.py", "line_start": 201}, result={"content": "y"})]
        )

        for idx, decision in enumerate((d1, d2, d3), start=1):
            assert decision.turn_number == idx
            assert decision.action == "continue"
            assert decision.is_repetitive is False
            assert decision.repeated_turns == []
            assert decision.system_message is None

    def test_happy_escalation_continue_then_warn_then_stop(self) -> None:
        tracker = TurnTracker(max_turns=10, max_repeats=3)
        call = [
            ToolCallRecord(
                tool_name="search_symbol",
                arguments={"path": "src", "symbol": "orders"},
                result={"total_matches": 2},
            )
        ]

        t1 = tracker.record_turn(call)
        assert t1.action == "continue"
        assert t1.system_message is None

        t2 = tracker.record_turn(call)
        assert t2.action == "warn"
        assert t2.is_repetitive is True
        assert t2.repeated_turns == [1]
        assert t2.system_message is not None
        assert "search_symbol" in t2.system_message
        assert "Do NOT repeat" in t2.system_message

        t3 = tracker.record_turn(call)
        assert t3.action == "stop"
        assert t3.is_repetitive is True
        assert t3.repeated_turns == [1, 2]
        assert t3.system_message is not None and "3 times" in t3.system_message

    def test_edge_parallel_tool_call_order_and_casing_still_detected(self) -> None:
        tracker = TurnTracker(max_turns=10, max_repeats=3)

        call_a = ToolCallRecord(tool_name="search_symbol", arguments={"symbol": "orders", "path": "src"})
        call_b = ToolCallRecord(tool_name="list_folder_content", arguments={"path": "tests"})

        t1 = tracker.record_turn([call_a, call_b])
        # Same calls in reversed order, different dict key order & uppercase tool name
        t2 = tracker.record_turn(
            [
                ToolCallRecord(tool_name="LIST_FOLDER_CONTENT", arguments={"path": "tests"}),
                ToolCallRecord(tool_name=" Search_Symbol ", arguments={"path": "src", "symbol": "orders"}),
            ]
        )

        assert t1.action == "continue"
        assert t2.action == "warn"
        assert t2.is_repetitive is True
        assert t2.repeated_turns == [1]

    def test_edge_consecutive_errors_warn_then_stop(self) -> None:
        tracker = TurnTracker(max_turns=10, max_repeats=3)

        t1 = tracker.record_turn(
            [ToolCallRecord(tool_name="read_file_slice", arguments={"path": "bad1.py"}, result={"error": "not found"}, is_error=True)]
        )
        t2 = tracker.record_turn(
            [ToolCallRecord(tool_name="read_file_slice", arguments={"path": "bad2.py"}, result={"error": "not found"}, is_error=True)]
        )
        t3 = tracker.record_turn(
            [ToolCallRecord(tool_name="read_file_slice", arguments={"path": "bad3.py"}, result={"error": "not found"}, is_error=True)]
        )

        assert t1.action == "continue"
        assert t2.action == "warn"
        assert t2.system_message is not None and "tool errors" in t2.system_message
        assert t3.action == "stop"

    def test_edge_error_streak_resets_after_successful_turn(self) -> None:
        tracker = TurnTracker(max_turns=10, max_repeats=3)

        tracker.record_turn([ToolCallRecord(tool_name="read_file_slice", arguments={"path": "bad1.py"}, is_error=True)])
        tracker.record_turn([ToolCallRecord(tool_name="read_file_slice", arguments={"path": "good.py"}, is_error=False)])
        t3 = tracker.record_turn([ToolCallRecord(tool_name="read_file_slice", arguments={"path": "bad2.py"}, is_error=True)])

        assert t3.action == "continue"

    def test_edge_max_turns_triggers_stop(self) -> None:
        tracker = TurnTracker(max_turns=2, max_repeats=3)

        t1 = tracker.record_turn([ToolCallRecord(tool_name="list_folder_content", arguments={"path": "."})])
        t2 = tracker.record_turn([ToolCallRecord(tool_name="list_folder_content", arguments={"path": "src"})])

        assert t1.action == "continue"
        assert t2.action == "stop"
        assert t2.system_message is not None and "max_turns" in t2.system_message

    def test_edge_separate_tracker_instances_do_not_share_state(self) -> None:
        call = [ToolCallRecord(tool_name="list_folder_content", arguments={"path": "."})]
        run1 = TurnTracker()
        run2 = TurnTracker()

        assert run1.record_turn(call).action == "continue"
        assert run2.record_turn(call).action == "continue"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_turns": 0},
            {"max_repeats": 1},
        ],
    )
    def test_sad_invalid_init_params_raise_value_error(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            TurnTracker(**kwargs)
