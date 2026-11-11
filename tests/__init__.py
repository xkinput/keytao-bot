"""Regression suites with the original collection order and duplicate coverage."""

import fnmatch
import importlib
from pathlib import Path
import unittest


# Legacy suites install different runtime stubs; their discovery order is significant.
_DISCOVERY = (
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.B10FlyKeyTests',)),
    ('tests.test_vote_regressions', ('tests.test_vote_regressions.PlusOneGuardTests', 'tests.test_vote_regressions.QQReceiptLinkTests')),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.B11BlockedToolLoopTests', 'tests.test_displace_shift_and_undo.B11ChangeSourceTests', 'tests.test_displace_shift_and_undo.B11ShiftTests', 'tests.test_displace_shift_and_undo.B11SourceShiftTests')),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B12SelectionTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.B13DefaultReadingTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B14MultiSelectionTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B15ChatTests', 'tests.test_add_and_select.B15PendingLookupTests')),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B16ItemFourTests', 'tests.test_receipts_and_display.B16ItemOneTests', 'tests.test_receipts_and_display.B16ItemThreeTests', 'tests.test_receipts_and_display.B16ItemTwoTests')),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B17ParserTests', 'tests.test_add_and_select.B17RestRouteTests', 'tests.test_add_and_select.B17ShiftPlanTests')),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B18RealToolTests', 'tests.test_add_and_select.B18ShiftValidationTests')),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B19PartialReplyBindingTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B20DisplayTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B21DisplayTests',)),
    ('tests.test_vote_regressions', ('tests.test_vote_regressions.B22StripperTests', 'tests.test_vote_regressions.B22VoteTests')),
    ('tests.test_commonness', ('tests.test_commonness.B23CommonnessTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B23DisplayTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.B24DisplaceTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.B25SessionTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.B26DisplayTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.B27ReadingTests',)),
    ('tests.test_routing_regressions', ('tests.test_semantic_b5_delete.B5DeleteTests', 'tests.test_routing_regressions.B5ReplySafetyTests')),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B6ReminderIntegrationTests', 'tests.test_routing_regressions.B6ReminderTests', 'tests.test_routing_regressions.B6SubmitTests', 'tests.test_routing_regressions.B7BlockedToolLoopTests')),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B7PhrasingRouteTests', 'tests.test_routing_regressions.B7QueryRouteTests')),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B7SelectionTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B8CancelTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.B8PresentWordsTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.AddKeepsSubmitTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.DiscoveryMessageIdTests', 'tests.test_receipts_and_display.MessageIdBindingTests')),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B9ParallelTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B9CancelTests', 'tests.test_routing_regressions.B9CandidateTests', 'tests.test_routing_regressions.B9WordQueryTests')),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.B9LookupTests', 'tests.test_routing_regressions.B9PronunciationEvidenceTests', 'tests.test_routing_regressions.B9RoutingSpeedTests')),
    ('tests.test_chat_word_discovery', ()),
    ('tests.test_community_vote', ()),
    ('tests.test_encoding_retry', ()),
    ('tests.test_github_sync_notification', ()),
    ('tests.test_glm_orchestrator', ()),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ExplicitAddHotfixTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.HotfixMultiwordTests',)),
    ('tests.test_image_input', ()),
    ('tests.test_isolation_fixes', ()),
    ('tests.test_keytao_encoding', ()),
    ('tests.test_llm_policy', ()),
    ('tests.test_memory_safety', ()),
    ('tests.test_operation_request', ()),
    ('tests.test_review_gate', ()),
    ('tests.test_add_and_select', ('tests.test_add_and_select.MultiwordRouteTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.S54RendererTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S54SelectionTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.FailureBoundaryTests',)),
    ('tests.test_search_regressions', ('tests.test_search_regressions.SearchBreakerTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.SinglePipelineTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.SingleReviewTests',)),
    ('tests.test_commonness', ('tests.test_commonness.S56CommonnessTests',)),
    ('tests.test_commonness', ('tests.test_commonness.CommonnessBindingTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S56AdvertisedClosureTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ExplicitCodeTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.GenericEvictionTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.S56SecurityTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.CompletedUndoTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.CompletedUndoReviewTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.OfferedAnswerClaimingTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.S57RigEvidenceReviewTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ExtraCodeTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S57OptionBoundaryTests', 'tests.test_add_and_select.S57OptionClaimTests', 'tests.test_add_and_select.S57OptionStructureTests')),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.SameCodeReorderTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ExtraCodeSecurityReviewTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.S58AdvertisementTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.BridgeTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.ExplicitDestinationTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.GenericProposalBridgeTests',)),
    ('tests.test_displace_shift_and_undo', ('tests.test_displace_shift_and_undo.MoveToCodeGrammarTests',)),
    ('tests.test_commonness', ('tests.test_commonness.CommonnessEvidenceTests',)),
    ('tests.test_commonness', ('tests.test_commonness.S59CommonnessRouteTests',)),
    ('tests.test_general_cap', ()),
    ('tests.test_tool_prompt', ()),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.ReceiptDeltaTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.ReceiptFinalizerTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.ReceiptRegressionTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.S60ScenarioTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.S61SubmitReplayTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S62AdversarialTests',)),
    ('tests.test_commonness', ('tests.test_commonness.S62CommonnessParserTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.DraftFlowTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S62IncidentTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ExplicitReadingReplayTests', 'tests.test_readings_and_flykeys.ReadingResolutionTests')),
    ('tests.test_add_and_select', ('tests.test_add_and_select.S62SelectionTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.S62DraftViewTests', 'tests.test_receipts_and_display.S62RepeatedRefusalTests', 'tests.test_receipts_and_display.S62SelectionDeliveryTests')),
    ('tests.test_commonness', ('tests.test_commonness.BccCommonnessTests',)),
    ('tests.test_commonness', ('tests.test_commonness.BccDeliveryTests',)),
    ('tests.test_commonness', ('tests.test_commonness.BccIngestTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.FreshSelectionTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.GeneralReplyTests',)),
    ('tests.test_commonness', ('tests.test_commonness.ExistingQueryTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.ExplicitSubmitGrammarTests', 'tests.test_add_and_select.ExplicitSubmitTests')),
    ('tests.test_commonness', ('tests.test_commonness.CommonnessRenderArchitectureTests', 'tests.test_commonness.CommonnessRenderTests')),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ReadingGrammarTests', 'tests.test_readings_and_flykeys.TranscriptTests')),
    ('tests.test_commonness', ('tests.test_commonness.S67RankingTests',)),
    ('tests.test_add_and_select', ('tests.test_add_and_select.ProposalTests', 'tests.test_add_and_select.SelectionTests')),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.FlyCodeTests', 'tests.test_readings_and_flykeys.FlyReviewTests', 'tests.test_readings_and_flykeys.FlyTranscriptTests', 'tests.test_readings_and_flykeys.LiveTicketTests', 'tests.test_readings_and_flykeys.PublishReceiptTests')),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.SmallTalkTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.QueryPrecedenceTests',)),
    ('tests.test_routing_regressions', ('tests.test_routing_regressions.StaleTicketReleaseTests',)),
    ('tests.test_receipts_and_display', ('tests.test_receipts_and_display.ReviewReasonTests',)),
    ('tests.test_readings_and_flykeys', ('tests.test_readings_and_flykeys.ReadingRereviewPersistenceTests',)),
    ('tests.test_security_fixes', ()),
    ('tests.test_semantic_add_route', ()),
    ('tests.test_semantic_assent', ()),
    ('tests.test_semantic_b5_delete', ()),
    ('tests.test_semantic_direct_exec', ()),
    ('tests.test_semantic_display_codes', ()),
    ('tests.test_semantic_model_first', ()),
    ('tests.test_semantic_parser', ()),
    ('tests.test_semantic_pending', ()),
    ('tests.test_semantic_pending_legacy', ()),
    ('tests.test_semantic_query_route', ()),
    ('tests.test_semantic_reorder_xcode', ()),
    ('tests.test_semantic_swap_confirmation', ()),
    ('tests.test_semantic_swap_route', ()),
    ('tests.test_sep19_commonness', ()),
    ('tests.test_sep19_general', ()),
    ('tests.test_sep19_pronunciation', ()),
    ('tests.test_sep19_reset', ()),
    ('tests.test_sep20_phrase', ()),
    ('tests.test_sep20_phrase_routing', ()),
    ('tests.test_skill_hardening', ()),
    ('tests.test_skills', ()),
    ('tests.test_state_machine', ()),
    ('tests.test_word_discovery', ()),
    ('tests.test_add_and_select', ('tests.test_add_and_select.SelectionCommandTests',)),
)


def load_tests(loader, _tests, pattern):
    pattern = pattern or "test*.py"
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    suite.addTests(loader.discover(str(root / "e2e"), pattern=pattern, top_level_dir=str(root)))
    for module_name, selectors in _DISCOVERY:
        if not fnmatch.fnmatch(module_name.rsplit(".", 1)[1] + ".py", pattern):
            continue
        try:
            module = importlib.import_module(module_name)
        except unittest.SkipTest as error:
            failed = unittest.loader._make_skipped_test(module_name, error, loader.suiteClass)
            suite.addTests(failed)
        except Exception:
            failed, message = unittest.loader._make_failed_import_test(module_name, loader.suiteClass)
            loader.errors.append(message)
            suite.addTests(failed)
        else:
            if selectors:
                suite.addTests(loader.loadTestsFromNames(selectors))
            else:
                suite.addTests(loader.loadTestsFromModule(module, pattern=pattern))
    return suite
