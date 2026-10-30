"""Project existing closed command grammars into a shadow operation request."""

from dataclasses import dataclass
from typing import Optional, Tuple

from . import authorization_grammar as grammar
from ..plugins import chat_routing as routing
from ..utils.explicit_code import parse_explicit_entry_code_request
from ..utils.pending_confirmation import parse_pending_candidate_selection
from ..utils.same_code_reorder import parse_same_code_reorder


@dataclass(frozen=True)
class OperationRequest:
    op: str = "none"
    operands: Tuple[str, ...] = ()
    codes: Tuple[str, ...] = ()
    submit_after: bool = False
    refers_to_pending: Optional[bool] = None
    frame: str = "unknown"
    source: str = "none"
    relation: str = ""
    code_selector: str = ""


def understand_message(
    message: str, pending_state: Optional[object] = None,
) -> OperationRequest:
    """Recognize a complete command before considering a live ticket answer."""
    def matched(parser, op, operands=(), codes=(), submit_after=False, pending=False):
        return OperationRequest(
            op=op, operands=tuple(value for value in operands if value),
            codes=tuple(value.lower() for value in codes if value),
            submit_after=submit_after, refers_to_pending=pending,
            frame="direct", source=f"fast_path:{parser}",
        )

    command = grammar.parse_entry_swap(message)
    if command is not None:
        return matched("parse_entry_swap", "swap", (command.first_word, command.second_word))
    command = grammar.parse_dictionary_delete_command(message)
    if command is not None:
        return matched("parse_dictionary_delete_command", "delete", (command.word,), (command.code,))
    command = grammar.parse_dictionary_recode_command(message)
    if command is not None:
        return matched("parse_dictionary_recode_command", "recode", (command.word,), (command.old_code, command.new_code))
    command = parse_same_code_reorder(message)
    if command is not None:
        return matched("parse_same_code_reorder", "reorder", (command.first_word, command.second_word), (command.first_code, command.second_code))
    command = routing._parse_word_list_reorder_command(message)
    if command is not None:
        return matched("_parse_word_list_reorder_command", "reorder", command.words)
    command = routing._parse_code_chain_reorder_command(message)
    if command is not None:
        return matched("_parse_code_chain_reorder_command", "reorder", command.focus_words, (command.code,))
    command = grammar.parse_entry_move_plan(message)
    if command is not None:
        return matched("parse_entry_move_plan", "move", tuple(move.word for move in command.moves), tuple(move.target_code for move in command.moves))
    command = grammar.parse_existing_entry_move(message)
    if command is not None:
        return matched("parse_existing_entry_move", "move", (command.word, command.named_occupant), (command.target_code,))
    command = grammar.parse_replace_at_code(message)
    if command is not None:
        return matched("parse_replace_at_code", "move", (command.old_word, command.new_word), (command.code,))
    command = parse_explicit_entry_code_request(message)
    if command is not None:
        return matched("parse_explicit_entry_code_request", "add", (command.word,), (command.code,), command.submit_after)
    command = grammar.explicit_complete_add_item(message)
    if command is not None:
        return matched("explicit_complete_add_item", "add", (command["word"],), (command["code"],), command["submitAfter"])
    command = grammar.parse_compound_eviction_add_plan(message)
    if command is not None:
        return matched("parse_compound_eviction_add_plan", "add", tuple(value for item in command.items for value in (item.word, item.named_occupant)), tuple(item.code for item in command.items))
    command = grammar.parse_eviction_modified_add(message)
    if command is not None:
        return matched("parse_eviction_modified_add", "add", (command.word, command.named_occupant), (command.code,))
    command = grammar._multi_add_authorization_contract(message)
    if command is not None and command.valid:
        return matched("_multi_add_authorization_contract", "add", tuple(item.word for item in command.clauses), tuple(item.code for item in command.clauses))
    command = routing.parse_draft_view_command(message)
    if command is not None:
        return matched("parse_draft_view_command", "draft_view")
    command = routing._structural_draft_management_intent(message)
    if command is not None and command.intent in {"draft_recall", "draft_clear"}:
        return matched("_structural_draft_management_intent", command.intent)
    if routing._is_explicit_draft_submit_request(message):
        return matched("_is_explicit_draft_submit_request", "draft_submit")

    # shortcut: batch_replace and draft_keep_only lack text-only operand parsers;
    # keep them unmapped until an existing closed grammar supplies the operands.
    if pending_state is not None:
        pending_word = getattr(pending_state, "word", "")
        if not pending_word and isinstance(pending_state, routing.PendingToolConfirm):
            if pending_state.function_name == "keytao_create_phrase":
                pending_word = pending_state.args.get("word", "")
            else:
                pairs = routing.pending_batch_display_pairs(pending_state)
                if len(pairs) == 1:
                    pending_word = pairs[0][0]
        command = grammar.parse_reviewed_selection_command(message)
        if command is not None:
            return matched(
                "parse_reviewed_selection_command", "select_candidate",
                tuple(word for word, _selector in command.pairs) + command.shift_words,
                tuple(selector for _word, selector in command.pairs if not selector.isdecimal()),
                "并提交" in command.action, pending=True,
            )
        command = parse_pending_candidate_selection(message)
        if command is not None:
            return matched("parse_pending_candidate_selection", "select_candidate", (pending_word,), command.codes, command.submit_after, pending=True)
        assent = routing._pending_assent_phrase_for_state(pending_state, message)
        if assent.cancel_requested:
            return matched("_pending_assent_phrase_for_state", "cancel", (pending_word,), pending=True)
        if assent.matched:
            return matched("_pending_assent_phrase_for_state", "confirm", (pending_word,), submit_after=assent.submit_after, pending=True)
    return OperationRequest()
