"""Topic regressions: commonness."""

import unittest as _stage_t_unittest
from tests.helpers import (
    AsyncMock,
    B18RealToolTests_setUp,
    B23CommonnessTests_candidate_review,
    B23CommonnessTests_seed_historical_chain,
    B23CommonnessTests_setUp,
    BccCommonnessTests_setUp,
    BccDeliveryTests_run_stage,
    BccDeliveryTests_setUp,
    BccIngestTests_cached_corpus,
    CHANNELS,
    COMMAND,
    CORPUS_MODES,
    CommonnessEvidenceTests_insert,
    CommonnessEvidenceTests_setUp,
    CommonnessRenderArchitectureTests_NON_RENDERERS,
    CommonnessRenderArchitectureTests_SITES,
    CommonnessRenderArchitectureTests_VERDICT_OWNERS,
    CommonnessRenderTests_assert_delivery,
    CommonnessRenderTests_local_comparison,
    CommonnessRenderTests_seed_both,
    CommonnessRenderTests_setUp,
    ExistingQueryTests_corpus,
    ExistingQueryTests_setUp,
    MESSAGES,
    PAIRS,
    Path,
    S56CommonnessTests_assess,
    S59CommonnessRouteTests_run_incident,
    S59CommonnessRouteTests_setUp,
    S67RankingTests_cells,
    S67RankingTests_reply,
    S67RankingTests_setUp,
    SQLiteConversationStateStore,
    SimpleNamespace,
    SkillsManager,
    TOOLS,
    _format_reviewed_add_prompt,
    advertised_batch_binding_pairs,
    advertised_command_suggestions,
    advertised_reply_contract,
    advertised_single_word_lookup_codes,
    already_existing_word_copy,
    ast,
    asyncio,
    b18_rest_real_tools_DraftHTTP,
    b18_rest_real_tools_FACTS,
    begin_turn_metrics,
    candidate_fixture,
    chat_render,
    closing,
    commonness_copy,
    commonness_query,
    copy,
    create_schema,
    current_turn_metrics,
    end_turn_metrics,
    format_frequency,
    harness,
    hashlib,
    ingest,
    io,
    json,
    lookup_fixture,
    lookup_word_commonness,
    occupied_review,
    os,
    parse_commonness_query,
    parse_eviction_modified_add,
    parse_explicit_code_request,
    parse_explicit_entry_code_request,
    patch,
    pending,
    pending_candidate,
    preview,
    product,
    protected_candidate_occupants,
    re,
    render,
    render_server_backed_single_word_lookup,
    review,
    s56_commonness_binding_chat,
    s56_commonness_binding_commands,
    s59_commonness_route_MESSAGE,
    s59_commonness_route_chat,
    s62_commonness_MESSAGE,
    s62_commonness_WORDS,
    s64_existing_query_chat,
    s64_existing_query_commands,
    same_code_reorder,
    seed_historical,
    single_review,
    sqlite3,
    subprocess,
    sys,
    tempfile,
    unittest,
    validate_commonness_words,
    zipfile,
)


# Source: test_b23_commonness.py
class B23CommonnessTests(unittest.IsolatedAsyncioTestCase):
    setUp = B23CommonnessTests_setUp

    candidate_review = B23CommonnessTests_candidate_review

    async def test_historical_incumbent_keeps_empty_slot_recommendation(self):
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")
        self.assertEqual(assessment["newCode"], "yygla")
        self.assertIn("2. yygla — ✅ 推荐（空位）", prompt)
        self.assertNotIn("需重排", prompt)
        self.assertIn("1,952", assessment["summary"])

    async def test_dictionary_counts_alone_never_reorder(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_dictionary_comparison_remains_available_to_read_only_lookup(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', NULL, NULL, 0)")
        comparison = await review.compare_word_commonness("羽衣甘蓝", "用过")
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        result = lookup_word_commonness(["用过", "羽衣甘蓝"])
        self.assertEqual([row["word"] for row in result["words"]], ["羽衣甘蓝", "用过"])
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_dictionary_counts_alone_cannot_reorder_current_chain(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', NULL, NULL, 0)")
        entries = [
            {"word": word, "code": "yygl", "type": "Phrase", "weight": index}
            for index, word in enumerate(("用过", "羽衣甘蓝"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_dictionary_counts_alone_cannot_approve_css_change(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO word_commonness VALUES (?, NULL, NULL, ?)",
                                   [("甲", 3), ("乙", 1)])
        http = b18_rest_real_tools_DraftHTTP()
        http.entries = [{"word": "乙", "code": "a", "type": "CSSSingle", "weight": 100}]
        with patch.object(review.http_client, "_keytao_client", http):
            result = await review.audit_draft_items(review.ReviewHttpConfig("offline", ""), [
                {"action": "Change", "word": "甲", "oldWord": "乙", "code": "a", "type": "CSSSingle"},
            ])
        comparison = result["reviewedWords"]["甲@CSSSingle"]["commonnessComparison"]
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        self.assertFalse(result["autoApprove"], result)
        self.assertIn("缺少足够的词库审查证据", result["issues"][0])
        self.assertEqual(result["approvedItems"], [])
        self.assertEqual(http.errors, [])
        self.assertEqual(http.writes, [])
        self.web.assert_not_awaited()

    async def test_dictionary_counts_alone_cannot_approve_advisory_priority(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO word_commonness VALUES (?, NULL, NULL, ?)",
                                   [("甲", 3), ("乙", 1)])
        http = b18_rest_real_tools_DraftHTTP()
        http.entries = [
            {"word": word, "code": code, "type": "CSSSingle", "weight": 100}
            for word, code in (("甲", "a"), ("乙", "b"))
        ]
        items = [
            {"action": action, "word": word, "code": code, "type": "CSSSingle"}
            for action, word, code in (
                ("Delete", "甲", "a"), ("Delete", "乙", "b"),
                ("Create", "甲", "b"), ("Create", "乙", "c"),
            )
        ]
        with patch.object(review.http_client, "_keytao_client", http):
            result = await review.audit_draft_items(review.ReviewHttpConfig("offline", ""), items)
        self.assertEqual(len(result["commonnessComparisons"]), 1)
        comparison = result["commonnessComparisons"][0]["result"]
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        self.assertEqual(len(result["approvedItems"]), 2, result)
        self.assertTrue(all(item.startswith("调码删除原位：") for item in result["approvedItems"]))
        self.assertFalse(result["autoApprove"], result)
        self.assertEqual(http.errors, [])
        self.assertEqual(http.writes, [])
        self.web.assert_not_awaited()

    async def test_new_corpus_frequency_cannot_displace_incumbent_without_modern_data(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=1000 WHERE word='羽衣甘蓝'")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_comparable_modern_corpus_frequency_still_reorders(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=1000 WHERE word='羽衣甘蓝'")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 100, 'v', 2)")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygl", prompt)
        self.assertEqual(assessment["verdict"], "front_more_common")
        self.assertEqual(assessment["decisionReason"], "frequency_ratio")
        self.assertIn("需重排", prompt)

    async def test_comparable_modern_bcc_frequency_still_reorders(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)", [
                ("多领域", "羽衣甘蓝", 1000, 10, 100),
                ("多领域", "用过", 100, 1, 1000),
            ])
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygl", prompt)
        self.assertEqual(assessment["decisionReason"], "bcc_frequency_ratio")
        self.assertIn("需重排", prompt)

    async def test_historical_frequency_alone_cannot_displace_incumbent(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET dictionary_presence_count=1")
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "羽衣甘蓝", 10000, 10000 * 1000000 / 835646749, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_historical_incumbent_without_dictionary_rows_stays_offline(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM word_commonness")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")

    async def test_incumbent_corpus_frequency_wins_without_dictionary_support(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 10, 'v', 0)")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")

    async def test_historical_advantage_cannot_override_close_modern_frequencies(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=10, dictionary_presence_count=1")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 12, 'v', 1)")
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "羽衣甘蓝", 10000, 10000 * 1000000 / 835646749, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_incumbent_bcc_frequency_is_cited_as_keep_order_evidence(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("多领域", "用过", 1000, 10, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")
        self.assertIn("多领域 1,000", assessment["summary"])
        self.assertIn("保留现有排序", assessment["summary"])
        self.assertNotIn("单边 BCC 收录不决定高低", assessment["summary"])

    seed_historical_chain = B23CommonnessTests_seed_historical_chain

    async def test_historical_only_later_entry_cannot_reorder_current_chain(self):
        self.seed_historical_chain()
        entries = [
            {"word": word, "code": "jsci", "type": "Phrase", "weight": index}
            for index, word in enumerate(("甲词", "乙词"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_one_sided_modern_bcc_cannot_reorder_current_chain(self):
        fixture = json.loads((Path(__file__).parents[1] / "e2e/fixtures/bcc/s63.json").read_text())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            for table in ("bcc_dataset", "bcc_frequency", "word_commonness"):
                connection.execute(f"DELETE FROM {table}")
            for table, rows in (("bcc_dataset", fixture["datasets"]),
                                ("bcc_frequency", fixture["rows"]),
                                ("word_commonness", fixture["jieba"])):
                for row in rows:
                    connection.execute(
                        f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})',
                        tuple(row.values()),
                    )
        entries = [
            {"word": word, "code": "qbsi", "type": "Phrase", "weight": index}
            for index, word in enumerate(("敲不死", "情报所"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_batch_create_collision_keeps_listed_order_with_history_only(self):
        self.seed_historical_chain()
        B18RealToolTests_setUp(self)
        self.enterContext(patch.dict(b18_rest_real_tools_FACTS, {
            word: ("jsci", ["jsci", "jscia"], "jia ci" if word == "甲词" else "yi ci")
            for word in ("甲词", "乙词")
        }))
        items = [{"word": word, "code": "jsci", "type": "Phrase", "action": "Create"}
                 for word in ("甲词", "乙词")]
        preview = await TOOLS.keytao_batch_add_to_draft(
            "qq", "b23-owner", items, batch_id=self.http.batch_id,
        )
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertTrue(preview.get("collisionReplanned"), preview)
        self.assertEqual(self.http.writes, [])
        result = await TOOLS.keytao_batch_add_to_draft(
            "qq", "b23-owner", preview["effectiveItems"], batch_id=self.http.batch_id,
            confirmed=True, expected_content_version=0, expected_warning_digest="a" * 64,
        )
        self.assertEqual(result.get("successCount"), 2, result)
        self.assertEqual(
            [(row["word"], row["code"]) for row in self.http.rows],
            [("甲词", "jsci"), ("乙词", "jscia")],
        )
        self.assertEqual(self.http.errors, [])
        self.web.assert_not_awaited()

# Source: test_s56_commonness.py
class S56CommonnessTests(unittest.IsolatedAsyncioTestCase):
    assess = S56CommonnessTests_assess

    async def test_single_occupied_candidate_is_compared_and_has_no_default(self):
        subject, compare = await self.assess(occupied_review(), "behind_more_common")
        compare.assert_awaited_once_with("鎗", "强")
        self.assertEqual("", subject["recommendedCode"])
        self.assertEqual("", subject["pronunciations"][0]["recommendedCode"])
        self.assertEqual("", subject["candidateOrderingAssessments"][0]["freeCode"])

    async def test_close_or_unknown_single_candidate_cannot_displace_by_default(self):
        for verdict in ("close", "not_enough_evidence"):
            with self.subTest(verdict=verdict):
                subject, _ = await self.assess(occupied_review(), verdict)
                self.assertEqual("", subject["recommendedCode"])

    async def test_weaker_word_occupant_retains_default_front_insert(self):
        subject, compare = await self.assess(
            occupied_review("单份", "蛋粉", "djff", phrase_type="Phrase", free="djffu"),
            "front_more_common",
        )
        compare.assert_awaited_once_with("单份", "蛋粉")
        self.assertEqual("djff", subject["recommendedCode"])
        self.assertEqual("djffu", subject["candidateOrderingAssessments"][0]["freeCode"])

    async def test_stronger_word_occupant_uses_free_slot(self):
        subject, _ = await self.assess(
            occupied_review("单份", "蛋粉", "djff", phrase_type="Phrase", free="djffu"),
            "behind_more_common",
        )
        self.assertEqual("djffu", subject["recommendedCode"])

    async def test_weaker_singleton_has_no_invented_empty_fallback(self):
        subject, _ = await self.assess(occupied_review(), "front_more_common")
        source = _format_reviewed_add_prompt(subject)
        self.assertEqual("qx", subject["recommendedCode"])
        self.assertIn("强」顺延", source)
        self.assertNotIn("不重排选", source)
        self.assertIn("回复「加入」", source)

    async def test_one_weaker_occupant_cannot_authorize_displacing_stronger_peer(self):
        subject = occupied_review(free="qxi")
        status = subject["pronunciations"][0]["candidateStatuses"][0]
        status["words"] = ["戕", "强"]
        status["phrases"] = [{"word": word, "code": "qx", "type": "Single"} for word in status["words"]]
        compare = AsyncMock(side_effect=[
            {"success": True, "verdict": "front_more_common"},
            {"success": True, "verdict": "behind_more_common"},
        ])
        with patch.object(review, "compare_word_commonness", compare):
            assessments = await review.assess_candidate_chain_commonness(subject)
        review.apply_candidate_ordering_recommendation(subject, assessments)
        self.assertEqual("qxi", subject["recommendedCode"])
        rendered = _format_reviewed_add_prompt(subject)
        self.assertNotIn("占 qx", rendered)
        self.assertIn("→ qxi（推荐）", rendered)

    async def test_explicit_code_occupant_uses_shared_comparator(self):
        compare = AsyncMock(return_value={"success": True, "verdict": "front_more_common"})
        with patch.object(review, "compare_word_commonness", compare):
            assessments = await review.assess_explicit_code_commonness(
                "单份", "djffuo", [{"word": "蛋粉", "type": "Phrase"}],
            )
        compare.assert_awaited_once_with("单份", "蛋粉")
        self.assertEqual((), protected_candidate_occupants(
            "单份", "djffuo", [("djffuo", True)], {"djffuo": ["蛋粉"]}, assessments,
        ))

    async def test_no_free_slot_renderer_keeps_review_and_advertises_named_eviction_only(self):
        subject, _ = await self.assess(occupied_review(), "behind_more_common")
        source = _format_reviewed_add_prompt(subject)
        self.assertIsNotNone(source)
        self.assertIn("推荐编码：暂无（现有候选均需点名顶替）", source)
        rendered = render_server_backed_single_word_lookup(
            "鎗", "", [("qx", True)], {"qx": ["强"]},
            reviewed_prompt=source, actionable_controls=True,
            ordering_assessments=subject["candidateOrderingAssessments"],
        )
        self.assertIn("读音 qiāng", rendered)
        self.assertIn("不弱于", rendered)
        self.assertIn("加入编码 qx+形码", rendered)
        self.assertNotIn("推荐编码：qx", rendered)
        self.assertNotIn("回复「加入」", rendered)
        self.assertEqual(("qx",), advertised_single_word_lookup_codes(rendered))
        commands = advertised_command_suggestions(rendered)
        self.assertEqual(("添加 鎗 qx，顶替 强",), commands)
        for command in commands:
            parsed = parse_eviction_modified_add(command)
            self.assertIsNotNone(parsed, command)
            self.assertEqual(("鎗", "qx", "强"), (parsed.word, parsed.code, parsed.named_occupant))

    def test_default_guard_requires_a_bound_verdict_for_every_occupant(self):
        assessments = [{"newWord": "鎗", "occupantWord": "强", "occupantCode": "qx", "verdict": "front_more_common"}]
        self.assertEqual(("戕",), protected_candidate_occupants(
            "鎗", "qx", [("qx", True)], {"qx": ["强", "戕"]}, assessments,
        ))
        self.assertEqual(("强",), protected_candidate_occupants(
            "鎗", "qxi", [("qxi", True)], {"qxi": ["强"]}, assessments,
        ))

    def test_incomplete_occupied_snapshot_never_authorizes_default_displacement(self):
        for occupants in ([], [""], None, "强"):
            with self.subTest(occupants=occupants):
                self.assertEqual(("",), protected_candidate_occupants(
                    "鎗", "qx", [("qx", True)], {"qx": occupants}, [],
                ))

# Source: test_s56_commonness_binding.py
class CommonnessBindingTests(unittest.TestCase):
    def test_occupied_only_discovery_survives_reload_and_whole_reply_binding(self):
        async def run(path):
            store = SQLiteConversationStateStore(db_path=path)
            key = harness.ConversationAddress.private("qq", "s56-binding")
            review = single_review()
            review["recommendedCode"] = ""
            review["pronunciations"][0]["recommendedCode"] = ""
            review["pronunciations"][0]["candidateStatuses"] = [{
                "code": "qx", "occupied": True, "words": ["强"], "label": "已有「强」",
                "phrases": [{"code": "qx", "word": "强", "type": "Single", "weight": 10}],
            }]
            review["candidateOrderingAssessments"] = [{
                "verdict": "behind_more_common", "newWord": "鎗", "occupantWord": "强",
                "occupantCode": "qx", "freeCode": "", "newCode": "",
            }]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(review)
                raise AssertionError(name)

            with patch.object(s56_commonness_binding_chat, "conversation_state_store", store), patch.object(
                s56_commonness_binding_chat, "call_tool_function", side_effect=tool,
            ), patch.object(s56_commonness_binding_chat, "_classify_simple_word_query_intent", AsyncMock(
                return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0),
            )), patch.object(s56_commonness_binding_commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await s56_commonness_binding_commands._try_handle_simple_single_word_query("鎗", "qq", "s56-binding", key)
                self.assertIn("推荐编码：暂无", reply)
                self.assertIn("审词：读音 qiāng", reply)
                reloaded = SQLiteConversationStateStore(db_path=path)
                record = reloaded.get_record(key)
                self.assertIsNotNone(record)
                self.assertEqual("", record.state.recommended_code)
                self.assertEqual("Single", record.state.phrase_type)
                self.assertEqual(review["candidateOrderingAssessments"], record.state.server_ordering_assessments)
                self.assertTrue(s56_commonness_binding_chat._advertised_reply_matches_live_record(reply, record), reply)
                self.assertEqual(reply, s56_commonness_binding_chat._enforce_advertised_reply_contract(reply, key))
                for command in advertised_command_suggestions(reply):
                    self.assertTrue(s56_commonness_binding_chat._chat_routing.message_authorizes_live_pending_mutation(command, record.state), command)
                for line in reply.splitlines():
                    if line.startswith("- 「"):
                        self.assertTrue(s56_commonness_binding_chat._chat_routing.message_authorizes_live_pending_mutation(line, record.state), line)
                self.assertFalse(s56_commonness_binding_chat._live_candidate_affordances_are_complete(
                    reply + "\n回复「加入」写入草稿，或回复「加入并提交」写入并提交。", record,
                ))
                self.assertTrue(s56_commonness_binding_chat._chat_routing.message_authorizes_live_pending_mutation("加入编码qxioio", record.state))
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))

# Source: test_s59_commonness_evidence.py
class CommonnessEvidenceTests(unittest.TestCase):
    setUp = CommonnessEvidenceTests_setUp

    insert = CommonnessEvidenceTests_insert

    def test_incident_words_use_review_rule_and_retain_absence(self):
        self.insert(
            ("耶博", 100, "n", 2),
            ("伊莎贝拉", 1000, "nr", 3),
            ("夜泊", 400, "v", 3),
        )
        result = lookup_word_commonness(["耶博", "伊莎贝拉", "夜泊", "一身本领"])
        self.assertTrue(result["success"])
        self.assertTrue(result["referenceAvailable"])
        self.assertEqual(result["method"], "offline_reference")
        self.assertEqual(
            [row["word"] for row in result["words"]],
            ["伊莎贝拉", "夜泊", "耶博", "一身本领"],
        )
        self.assertEqual([row["rank"] for row in result["words"]], [1, 2, 3, None])
        self.assertEqual([row["verdict"] for row in result["words"]], ["ranked"] * 3 + ["unknown"])
        self.assertEqual(result["words"][-1]["dictionaryPresenceCount"], None)
        self.assertEqual(result["words"][-1]["corpusFrequency"], None)
        self.assertFalse(result["words"][-1]["known"])
        comparison = next(item for item in result["comparisons"] if
                          item["frontWord"] == "耶博" and item["behindWord"] == "夜泊")
        expected = review._compare_reference_commonness(
            "耶博", "夜泊", review._query_commonness_reference("耶博"),
            review._query_commonness_reference("夜泊"),
        )
        self.assertEqual(comparison, expected)
        self.assertIn(comparison["summary"], review.current_commonness_evidence())
        self.web.assert_not_called()

    def test_close_is_not_sorted_by_weighted_score(self):
        self.insert(("低频", 100, "n", 1), ("高频", 150, "n", 3))
        result = lookup_word_commonness(["低频", "高频"])
        self.assertEqual([row["word"] for row in result["words"]], ["低频", "高频"])
        self.assertEqual([row["verdict"] for row in result["words"]], ["close", "close"])
        self.assertEqual(result["comparisons"][0]["verdict"], "close")

    def test_equal_frequency_and_unknown_relation_remain_explicit(self):
        self.insert(("相同甲", 100, "n", 1), ("相同乙", 100, "n", 1))
        result = lookup_word_commonness(["相同甲", "相同乙"])
        self.assertEqual([row["verdict"] for row in result["words"]], ["close", "close"])
        self.insert(("词典甲", None, None, 1), ("词典乙", None, None, 2))
        result = lookup_word_commonness(["词典甲", "词典乙"])
        self.assertTrue(all(row["known"] for row in result["words"]))
        self.assertEqual([row["verdict"] for row in result["words"]], ["unknown", "unknown"])
        self.assertEqual(result["comparisons"][0]["verdict"], "not_enough_evidence")

    def test_dictionary_only_and_valid_zero_presence_keep_missing_distinct(self):
        self.insert(("语料词", 100, "n", 0), ("词典词", None, None, 3))
        result = lookup_word_commonness(["语料词", "词典词", "未知词"])
        by_word = {row["word"]: row for row in result["words"]}
        self.assertEqual(by_word["语料词"]["dictionaryPresenceCount"], 0)
        self.assertTrue(by_word["词典词"]["known"])
        self.assertIsNone(by_word["词典词"]["corpusFrequency"])
        self.assertIsNone(by_word["未知词"]["dictionaryPresenceCount"])
        self.assertEqual(result["words"][0]["word"], "词典词")

    def test_zero_frequency_is_invalid_reference_data_not_absence_coerced_to_zero(self):
        self.insert(("零频", 0, "n", 1))
        row = lookup_word_commonness(["零频"])["words"][0]
        self.assertFalse(row["known"])
        self.assertFalse(row["referenceAvailable"])
        self.assertIsNone(row["corpusFrequency"])
        self.assertIsNone(row["dictionaryPresenceCount"])

    def test_absent_and_unavailable_database_never_use_web(self):
        for path, expected_available in ((self.path, True), (self.path.with_name("missing.db"), False)):
            with self.subTest(path=path), patch.object(review, "reference_db_path", return_value=path):
                result = lookup_word_commonness(["不存在甲", "不存在乙"])
                self.assertEqual(result["referenceAvailable"], expected_available)
                self.assertTrue(all(not row["known"] for row in result["words"]))
                self.assertTrue(all(row["rank"] is None for row in result["words"]))
        self.web.assert_not_called()

    def test_bounded_input_validation_precedes_database_lookup(self):
        self.assertEqual(validate_commonness_words(["  词  "]), ["词"])
        self.assertEqual(len(validate_commonness_words([f"词{i}" for i in range(12)])), 12)
        invalid = (None, "甲乙", [], [""], [None], ["甲", "甲"], ["甲", " 甲 "],
                   ["x" * 65], ["甲\n乙"], [f"词{i}" for i in range(13)])
        with patch.object(review, "_query_commonness_reference", side_effect=AssertionError("must validate first")):
            for words in invalid:
                with self.subTest(words=words), self.assertRaises(ValueError):
                    lookup_word_commonness(words)
        with self.assertRaises(ValueError):
            validate_commonness_words(["单词"], min_words=2)

    def test_shared_stable_order_matches_review_chain_and_detects_cycles(self):
        self.insert(("甲词", 100, "n", 2), ("乙词", 150, "n", 2), ("丙词", 400, "n", 2))
        words = ["甲词", "乙词", "丙词"]
        result = lookup_word_commonness(words)
        chain = asyncio.run(review.rank_code_chain_by_commonness([
            {"word": word, "weight": index + 1} for index, word in enumerate(words)
        ]))
        self.assertEqual([row["word"] for row in result["words"]],
                         [row["word"] for row in chain["proposedOrder"]])
        self.assertIsNone(review._stable_commonness_order(words, {
            "甲词": {"乙词"}, "乙词": {"丙词"}, "丙词": {"甲词"},
        }))

    def test_conflicting_evidence_retains_all_rows_without_claiming_ranks(self):
        self.insert(("语料领先", 100, "n", 0), ("两类收录", 50, "n", 3),
                    ("词典领先", None, None, 3))
        words = ["语料领先", "两类收录", "词典领先", "未知词"]
        result = lookup_word_commonness(words)
        self.assertEqual(result["ordering"], "conflicting_evidence")
        self.assertEqual([row["word"] for row in result["words"]], words)
        self.assertTrue(all(row["rank"] is None for row in result["words"]))
        self.assertTrue(all(row["verdict"] == "unknown" for row in result["words"]))
        self.assertEqual([row["corpusFrequency"] for row in result["words"]], [100, 50, None, None])

# Source: test_s59_commonness_route.py
class S59CommonnessRouteTests(unittest.TestCase):
    setUp = S59CommonnessRouteTests_setUp

    def test_closed_grammar_and_bounds(self):
        for message in (s59_commonness_route_MESSAGE, "词频排序 耶博、伊莎贝拉，夜泊 一身本领",
                        "按常用度排序：耶博和夜泊", "哪个更常用：耶博 夜泊",
                        "常用度对比：耶博、夜泊", "耶博 和 夜泊 哪个更常用？"):
            with self.subTest(message=message):
                self.assertIsNotNone(self.query.parse_commonness_query(message))
        self.assertEqual(self.query.parse_commonness_query("常用度对比：和平 和 夜泊"), ("和平", "夜泊"))
        self.assertEqual(self.query.parse_commonness_query("词频排序：夜泊和耶博、伊莎贝拉"), ("夜泊", "耶博", "伊莎贝拉"))
        self.assertEqual(self.query.parse_commonness_query("词频排序：共和国 夜泊"), ("共和国", "夜泊"))
        self.assertEqual(self.query.parse_commonness_query("哪个更常用：添加 删除"), ("添加", "删除"))
        for message in ("词频排序 耶博", "词频排序 耶博 耶博", "他说词频排序：耶博 夜泊",
                        "不要词频排序：耶博 夜泊", "词频排序 耶博 夜泊；删除夜泊",
                        "词频排序 " + " ".join(chr(0x4e00 + i) for i in range(13))):
            self.assertIsNone(self.query.parse_commonness_query(message), message)
        self.assertEqual(len(self.query.parse_commonness_query("词频排序 " + " ".join(chr(0x4e00 + i) for i in range(12)))), 12)

    def test_incident_enters_zero_model_stage_before_classifiers(self):
        names = [stage.__name__ for stage in s59_commonness_route_chat.STAGES]
        self.assertIn("_stage_handle_commonness_query", names)
        self.assertLess(names.index("_stage_handle_commonness_query"),
                        names.index("_stage_resolve_current_pending_scope"))

    run_incident = S59CommonnessRouteTests_run_incident

    def test_s59_fixture_scenario_table_affordance_and_delivery_closure(self):
        async def run():
            reply, calls = await self.run_incident()
            self.assertIn("名次 | 词 | 语料频次 | 词典收录 | 数据收录", reply)
            self.assertIn("1 | 伊莎贝拉 | jieba 参考 1,000 | 3", reply)
            self.assertIn("2 | 夜泊 | jieba 参考 400 | 3", reply)
            self.assertIn("3 | 耶博 | jieba 参考 100 | 2", reply)
            self.assertIn("— | 一身本领 | — | — | 无数据", reply)
            suggestions = self.query.advertised_command_suggestions(reply)
            self.assertEqual(suggestions, ('把 夜泊 yebo 排到 耶博 yebo 前面',))
            parsed = self.query.parse_same_code_reorder(suggestions[0])
            self.assertEqual((parsed.first_word, parsed.second_word, parsed.first_code), ("夜泊", "耶博", "yebo"))
            self.assertTrue(self.query.matches_commonness_delivery(reply, self.owner, s59_commonness_route_MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply + "\n可发送「删除耶博」", self.owner, s59_commonness_route_MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply, harness.ConversationAddress.private("qq", "other"), s59_commonness_route_MESSAGE))
            self.assertFalse(self.query.matches_commonness_delivery(reply, self.owner, "词频排序 甲 乙"))
            self.query.reset_commonness_delivery()
            self.assertFalse(self.query.matches_commonness_delivery(reply, self.owner, s59_commonness_route_MESSAGE))
            self.assertTrue(all(name in {"keytao_lookup_by_words_batch", "keytao_encode"} for name, _ in calls))
            return reply
        self.last_reply = asyncio.run(run())

    def test_absent_word_uses_encoding_and_s50_parser(self):
        async def run():
            reply, calls = await self.run_incident(missing=True)
            suggestions = self.query.advertised_command_suggestions(reply)
            self.assertIn("把 夜泊 放在 耶博 前面", suggestions)
            parsed = self.query.parse_eviction_modified_add(suggestions[0])
            self.assertEqual((parsed.word, parsed.named_occupant), ("夜泊", "耶博"))
            self.assertIn(("keytao_encode", {"word": "夜泊"}), calls)
        asyncio.run(run())

    def test_no_shared_chain_or_failed_lookup_keeps_table_without_affordance(self):
        async def run():
            for options in ({"shared": False}, {"lookup_failure": True}):
                reply, _calls = await self.run_incident(**options)
                self.assertIn("夜泊 | jieba 参考 400 | 3", reply)
                self.assertFalse(self.query.advertised_command_suggestions(reply))
        asyncio.run(run())

    def test_ambiguous_same_code_types_and_divergent_suffixes_do_not_advertise(self):
        async def run():
            for ambiguous in (True, False):
                async def read(name, args):
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [
                            {"word": word, "phrases": [
                                {"word": word, "code": "abcd" if ambiguous else code,
                                 "type": kind, "weight": 100}
                                for kind in (("Phrase", "CSS") if ambiguous else ("Phrase",))
                            ]}
                            for word, code in (("甲词", "abcxa"), ("乙词", "abcya"))
                        ]}
                    self.assertEqual(name, "keytao_encode")
                    return {"success": True, "word": args["word"], "codes": [
                        "abc", "abcxa" if args["word"] == "甲词" else "abcya",
                    ]}
                self.assertEqual(await self.query._placement_commands(
                    ("甲词", "乙词"), [{"frontWord": "甲词", "behindWord": "乙词", "verdict": "front_more_common"}], read,
                ), ())
        asyncio.run(run())

    def test_affordance_timeout_never_loses_local_ranking(self):
        async def run():
            with patch.object(self.query, "_placement_commands", side_effect=TimeoutError):
                reply, _calls = await self.run_incident()
            self.assertIn("夜泊 | jieba 参考 400 | 3", reply)
            self.assertIn("一身本领 | — | — | 无数据", reply)
            self.assertFalse(self.query.advertised_command_suggestions(reply))
        asyncio.run(run())

    def test_internal_fields_cannot_cross_actual_send_boundary(self):
        async def run():
            bot = SimpleNamespace(send=AsyncMock())
            for field in ("suggestedCommand", "blockReason", "boundTarget", "policyBlocked", "requiresTextFollowUp", "keytao_word_commonness"):
                raw = f"目前没有 {field}，请补充数据。"
                await s59_commonness_route_chat._send_event_response(bot, object(), "s59", None, raw, emit_metrics=False)
                sent = bot.send.call_args.kwargs["message"]
                self.assertNotIn(field.lower(), sent.lower())
                self.assertTrue(sent)
                with self.assertRaises(ValueError):
                    s59_commonness_route_chat._chat_render._assert_plain_user_facing_reply(raw)
        asyncio.run(run())

    def test_post_answer_augmentation_cannot_spend_a_third_model_call(self):
        from keytao_bot.utils.observability import begin_turn_metrics, current_turn_metrics, end_turn_metrics

        async def run():
            token = begin_turn_metrics("qq", "private")
            try:
                current_turn_metrics().model_calls = 2
                ctx = SimpleNamespace(response="现有回答", generic_command_intent=SimpleNamespace(intent="none"),
                                      normalized_message_text="日常问题")
                with patch.object(s59_commonness_route_chat, "_get_simple_word_query_words", side_effect=AssertionError("third model call")):
                    self.assertFalse(await s59_commonness_route_chat._stage_augment_word_query(ctx))
                self.assertEqual(ctx.response, "现有回答")
            finally:
                end_turn_metrics(token)
        asyncio.run(run())

# Source: test_s62_commonness.py
class S62CommonnessParserTests(unittest.TestCase):
    def test_incident_suffix_returns_only_the_six_explicit_words(self):
        self.assertEqual(parse_commonness_query(s62_commonness_MESSAGE), s62_commonness_WORDS)

    def test_negative_reported_and_mixed_write_requests_are_not_read_only_queries(self):
        for message in (
            "不要" + s62_commonness_MESSAGE,
            "别" + s62_commonness_MESSAGE,
            s62_commonness_MESSAGE.replace("排列", "不要排列"),
            "他说" + s62_commonness_MESSAGE,
            "他说：" + s62_commonness_MESSAGE,
            "「" + s62_commonness_MESSAGE + "」",
            s62_commonness_MESSAGE + "，然后删除加辆",
            s62_commonness_MESSAGE + "并加入草稿",
            "加量，加梁；删除加辆，排列以上词的使用频率",
            "加量，加梁，删除加辆，排列以上词的使用频率",
            "加量，加梁，加入草稿，排列以上词的使用频率",
            "加量，加梁，修改加辆，排列以上词的使用频率",
            "加量，加梁，并提交，排列以上词的使用频率",
            "加量，加梁，把加辆排到加量前面，排列以上词的使用频率",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_commonness_query(message))

    def test_suffix_keeps_literal_word_and_count_bounds(self):
        self.assertEqual(parse_commonness_query(s62_commonness_MESSAGE + "？"), s62_commonness_WORDS)
        self.assertEqual(parse_commonness_query(s62_commonness_MESSAGE.replace("排列", "请排列")), s62_commonness_WORDS)
        twelve = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", "子", "丑")
        suffix = "，排列以上词的使用频率"
        self.assertEqual(parse_commonness_query("、".join(twelve) + suffix), twelve)
        self.assertEqual(parse_commonness_query("甲" * 32 + "，乙" + suffix), ("甲" * 32, "乙"))
        for body in ("加量", "加量，加量", "加量，ASCII", "甲" * 33 + "，乙", "、".join((*twelve, "寅"))):
            with self.subTest(body=body):
                self.assertIsNone(parse_commonness_query(body + suffix))

    def test_existing_comparison_forms_keep_their_literal_words(self):
        for message, words in (
            ("词频排序：夜泊和耶博、伊莎贝拉", ("夜泊", "耶博", "伊莎贝拉")),
            ("按常用度排序：耶博和夜泊", ("耶博", "夜泊")),
            ("耶博 和 夜泊 哪个更常用？", ("耶博", "夜泊")),
            ("哪个更常用：添加 删除", ("添加", "删除")),
            ("常用度对比：和平 和 夜泊", ("和平", "夜泊")),
        ):
            with self.subTest(message=message):
                self.assertEqual(parse_commonness_query(message), words)

# Source: test_s63_bcc.py
class BccCommonnessTests(unittest.TestCase):
    setUp = BccCommonnessTests_setUp

    def test_real_pairs_decide_in_both_directions(self):
        for (common, rare), expected in zip(PAIRS, ('not_enough_evidence', 'not_enough_evidence', 'front_more_common')):
            with self.subTest(common=common):
                result = asyncio.run(review.compare_word_commonness(common, rare))
                self.assertEqual(result['verdict'], expected)
                self.assertFalse(result['decisionReason'].startswith('bcc_'))
                inverse = asyncio.run(review.compare_word_commonness(rare, common))
                self.assertEqual(inverse['verdict'], 'behind_more_common' if expected == 'front_more_common' else expected)
                if expected == 'front_more_common':
                    self.assertEqual(result['summary'], inverse['summary'])
        self.web.assert_not_called()

    def test_tool_exposes_history_without_changing_modern_headline(self):
        before = lookup_word_commonness(['情报所'])['words'][0]
        seed_historical(self.db, counts={'情报所': (9000, 1)})
        after = lookup_word_commonness(['情报所'])['words'][0]
        self.assertEqual(after['bcc']['perMillion'], before['bcc']['perMillion'])
        self.assertEqual(after['score'], before['score'])
        history = {row['channel']: row for row in after['bcc']['historicalChannels']}
        self.assertEqual(history['古代汉语']['count'], 9000)
        self.assertEqual(history['古代汉语']['perMillion'], 900000)
        self.assertIn('近代汉语', history)

    def test_historical_tiebreak_is_last_resort_and_explicit_in_copy(self):
        seed_historical(self.db, counts={'古例甲': (1234, 1), '古例乙': (12, 50)})
        result = asyncio.run(review.compare_word_commonness('古例甲', '古例乙'))
        self.assertEqual(result['verdict'], 'front_more_common')
        self.assertEqual(result['decisionReason'], 'bcc_historical_classical_frequency_ratio')
        self.assertIn('现代四频道均未收录', result['summary'])
        self.assertIn('词典与 jieba 无明确方向', result['summary'])
        self.assertIn('按古代汉语频次', result['summary'])
        self.assertIn('1,234', result['summary'])
        self.assertIn('12', result['summary'])
        inverse = asyncio.run(review.compare_word_commonness('古例乙', '古例甲'))
        self.assertEqual(inverse['verdict'], 'behind_more_common')
        self.assertEqual(inverse['summary'], result['summary'])
        ranked = lookup_word_commonness(['古例乙', '古例甲'])
        self.assertEqual([row['word'] for row in ranked['words']], ['古例甲', '古例乙'])
        self.assertIsNone(ranked['words'][0]['bcc']['perMillion'])
        self.web.assert_not_called()

    def test_history_cannot_win_a_short_code_against_modern_attestation(self):
        seed_historical(self.db, counts={'古例甲': (9999, 1), '情报所': (1, 100)})
        for words in (('古例甲', '情报所'), ('情报所', '古例甲')):
            result = asyncio.run(review.compare_word_commonness(*words))
            self.assertEqual(result['verdict'], 'not_enough_evidence')
            self.assertNotIn('historical', result['decisionReason'])
            self.assertNotIn('古代汉语', result['summary'])
            chain = asyncio.run(review.rank_code_chain_by_commonness([
                {'word': word, 'code': 'qbsi', 'type': 'Phrase', 'weight': index}
                for index, word in enumerate(words)]))
            self.assertEqual(chain['status'], 'ask')
            modern_line = next(line for line in chain['evidenceLines'] if '「情报所」' in line)
            historical_line = next(line for line in chain['evidenceLines'] if '「古例甲」' in line)
            self.assertNotIn('古代汉语', modern_line)
            self.assertIn('古代汉语', historical_line)
        placement = asyncio.run(review.assess_explicit_code_commonness(
            '古例甲', 'qbsi', [{'word': '情报所', 'type': 'Phrase'}]))
        self.assertEqual(placement[0]['verdict'], 'not_enough_evidence')

    def test_history_never_overrules_jieba_dictionary_or_modern_close(self):
        for frequencies, presences, modern, expected in (
            ((20, 200), (1, 1), False, 'behind_more_common'),
            ((None, None), (0, 2), False, 'behind_more_common'),
            ((10, 12), (0, 2), False, 'close'),
            ((None, None), (0, 0), True, 'close'),
            ((10, 12), (1, 1), False, 'front_more_common'),
        ):
            with self.subTest(frequencies=frequencies, presences=presences, modern=modern):
                seed_historical(self.db, counts={'古例甲': (1234, 1), '古例乙': (12, 50)})
                with closing(sqlite3.connect(self.db)) as connection, connection:
                    connection.executemany('INSERT OR REPLACE INTO word_commonness VALUES (?, ?, ?, ?)',
                        [(word, frequency, 'n', presence) for word, frequency, presence in
                         zip(('古例甲', '古例乙'), frequencies, presences)])
                    connection.execute("DELETE FROM bcc_frequency WHERE dataset='news_total_word_freq.txt' AND token IN ('古例甲', '古例乙')")
                    if modern:
                        connection.executemany('INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)',
                            [('news_total_word_freq.txt', word, count, count / 1000, 100)
                             for word, count in (('古例甲', 10), ('古例乙', 12))])
                result = asyncio.run(review.compare_word_commonness('古例甲', '古例乙'))
                self.assertEqual(result['verdict'], expected)
                self.assertEqual(result['decisionReason'].startswith('bcc_historical_'), expected == 'front_more_common')

    def test_historical_absence_incomplete_modern_and_unlike_units_defer(self):
        seed_historical(self.db, counts={'古例甲': (1234, 1)})
        result = lookup_word_commonness(['古例甲', '古例乙'])
        self.assertEqual(result['comparisons'][0]['verdict'], 'not_enough_evidence')
        self.assertIsNone(result['words'][0]['bcc']['perMillion'])
        seed_historical(self.db, counts={'古例甲': (1234, 1), '古例乙': (12, 50)})
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_dataset WHERE filename='dialogue_word_freq.txt'")
        result = lookup_word_commonness(['古例甲', '古例乙'])
        self.assertEqual(result['comparisons'][0]['verdict'], 'not_enough_evidence')
        self.assertFalse(result['words'][0]['bcc']['available'])

    def test_historical_cross_type_uses_own_table_rank_and_later_period_first(self):
        seed_historical(self.db, counts={'古例甲': (20, 1)}, total=110, rows=10)
        seed_historical(self.db, kind='char', counts={'龖': (900, 1)}, total=1000, rows=2)
        result = asyncio.run(review.compare_word_commonness('龖', '古例甲'))
        self.assertEqual(result['verdict'], 'behind_more_common')
        self.assertIn('relative_rank', result['decisionReason'])
        self.assertIn('「古例甲」前 10.00% vs 「龖」前 50.00%', result['summary'])
        self.assertGreater(result['front']['bcc']['historicalChannels'][0]['perMillion'],
                           result['behind']['bcc']['historicalChannels'][0]['perMillion'] * 2)
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE bcc_frequency SET frequency_rank=NULL WHERE token='龖'")
        self.assertEqual(lookup_word_commonness(['龖', '古例甲'])['comparisons'][0]['verdict'], 'not_enough_evidence')
        seed_historical(self.db, counts={'古例甲': (1234, 1), '古例乙': (12, 50)})
        seed_historical(self.db, channel='近代汉语', counts={'古例甲': (12, 50), '古例乙': (1234, 1)})
        result = asyncio.run(review.compare_word_commonness('古例甲', '古例乙'))
        self.assertEqual(result['verdict'], 'behind_more_common')
        self.assertIn('early_modern', result['decisionReason'])
        self.assertIn('按近代汉语频次', result['summary'])

    def test_unknown_coinages_restore_web_fallback_with_bcc_installed(self):
        self.web.side_effect = None
        self.web.return_value = {'success': True, 'score': 0, 'signals': {}}
        result = asyncio.run(review.compare_word_commonness('量子薄荷鸽跃器', '星云奶酪折叠喵'))
        self.assertEqual(result['verdict'], 'not_enough_evidence')
        self.assertIn('信号不足', result['summary'])
        self.assertFalse(lookup_word_commonness(['量子薄荷鸽跃器'])['words'][0]['known'])
        self.assertEqual(self.web.await_count, 2)
        asyncio.run(review.estimate_word_commonness('量子薄荷鸽跃器'))
        self.assertEqual(self.web.await_count, 3)

    def test_single_character_uses_char_counts_even_when_word_counts_exist(self):
        row = lookup_word_commonness(['的'])['words'][0]
        self.assertEqual(row['bcc']['tokenType'], 'char')
        balanced = row['bcc']['channels'][0]
        self.assertEqual(balanced['count'], 22900677)
        self.assertAlmostEqual(balanced['perMillion'], 35539.83070929854)
        self.assertNotEqual(balanced['count'], 22388585)
        history = {item['channel']: item for item in row['bcc']['historicalChannels']}
        self.assertEqual(history['近代汉语']['count'], 5610065)
        self.assertNotEqual(history['近代汉语']['count'], 5230487)

    def test_small_real_signal_cannot_defeat_an_absent_word(self):
        result = asyncio.run(review.compare_word_commonness('一一化', '量子薄荷鸽跃器'))
        self.assertEqual(result['verdict'], 'not_enough_evidence')
        self.assertEqual(result['front']['bcc']['channels'][0]['count'], 6)
        self.assertEqual(result['front']['reference']['dictionaryPresenceCount'], 0)

    def test_review_reproductions_preserve_dictionary_and_jieba_fallback(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany('INSERT OR REPLACE INTO word_commonness VALUES (?, ?, ?, ?)',
                [('户晨风', None, 'nr', 2), ('沃集鲜', 500, 'n', 0)])
        for word, expected in (('户晨风', 'front_more_common'), ('沃集鲜', 'not_enough_evidence')):
            with self.subTest(word=word):
                result = asyncio.run(review.compare_word_commonness(word, '一一化'))
                self.assertEqual(result['verdict'], expected)
                self.assertFalse(result['decisionReason'].startswith('bcc_'))
                inverse = asyncio.run(review.compare_word_commonness('一一化', word))
                self.assertNotEqual(inverse['verdict'], 'front_more_common')
                self.assertIsNone(result['front']['bcc']['perMillion'])

    def test_one_sided_bcc_preserves_the_complete_legacy_decision(self):
        hit = review._query_commonness_reference('一一化')['bcc']
        absent = review._query_commonness_reference('户晨风')['bcc']
        for frequencies, presences in product(product((None, 8, 10, 500), repeat=2), product(range(4), repeat=2)):
            legacy = [dict(available=True, attested=frequency is not None or presence > 0,
                           corpusFrequency=frequency, dictionaryPresenceCount=presence)
                      for frequency, presence in zip(frequencies, presences)]
            expected = review._compare_reference_commonness('甲词', '乙词', *legacy)
            for bccs in ((hit, absent), (absent, hit)):
                with self.subTest(frequencies=frequencies, presences=presences, hit=bccs[0]['attested']):
                    actual = review._compare_reference_commonness('甲词', '乙词', *[
                        dict(item, attested=item['attested'] or bcc['attested'], bcc=bcc)
                        for item, bcc in zip(legacy, bccs)])
                    self.assertEqual((actual['verdict'], actual['decisionReason']),
                                     (expected['verdict'], expected['decisionReason']))

    def test_cross_type_tiebreak_copy_uses_the_actual_balanced_ranks(self):
        char = review._query_commonness_reference('的')
        word = review._query_commonness_reference('五角星')
        for item, balanced in ((char, .5), (word, .1)):
            item['bcc']['rankFraction'] = .1
            for row in item['bcc']['channels']:
                row['rankFraction'] = balanced if row['channel'] == '多领域' else .1
        result = review._compare_reference_commonness('的', '五角星', char, word)
        self.assertEqual(result['verdict'], 'behind_more_common')
        self.assertIn('最高频道信号相同，采用多领域比较', result['summary'])
        self.assertIn('前 10.00% vs 前 50.00%', result['summary'])

    def test_bcc_attestation_does_not_block_either_semantic_override_entry(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('一一化', NULL, 'n', 2)")
        semantic = {'preSubmitAudit': {'semanticContextAutoPassItems': [{
            'word': '情报所', 'assessment': {'accepted': True, 'confidence': 0.99,
                'meaning': 'fixture modern use', 'nonObscurity': {
                    'route': 'common_characters_and_llm',
                    'characterReferences': [{'corpusFrequency': 10000}] * 3}}}]}}
        pair = {'newWord': '情报所', 'occupantWord': '一一化'}
        comparison = asyncio.run(review.compare_word_commonness('情报所', '一一化'))
        self.assertTrue(comparison['decisionReason'].startswith('bcc_'))
        overridden = review._modern_semantic_commonness_override(semantic, pair, comparison)
        self.assertEqual(overridden['decisionReason'], 'modern_semantic_vs_dictionary_dominated')
        for words in (('一一化', '情报所'), ('情报所', '一一化')):
            loader = AsyncMock(side_effect=lambda word: semantic if word == '情报所' else {})
            result = asyncio.run(review.rank_code_chain_by_commonness([
                {'word': word, 'code': 'qbsi', 'type': 'Phrase', 'weight': index}
                for index, word in enumerate(words)], semantic_review_loader=loader))
            self.assertEqual(result['comparisons'][0]['decisionReason'], 'modern_semantic_vs_dictionary_dominated')
            self.assertEqual(result['proposedOrder'][0]['word'], '情报所')
            loader.assert_awaited_once_with('情报所')

    def test_historical_channels_cannot_influence_a_modern_verdict(self):
        before = asyncio.run(review.compare_word_commonness('情报所', '敲不死'))
        for channel in ('古代汉语', '近代汉语'):
            seed_historical(self.db, channel=channel, counts={'敲不死': (10**12, 1)}, total=10**12)
        after = asyncio.run(review.compare_word_commonness('情报所', '敲不死'))
        for key in ('verdict', 'decisionReason', 'summary', 'scoreDelta'):
            self.assertEqual(after[key], before[key])
        for side in ('front', 'behind'):
            for key in ('perMillion', 'rankFraction', 'attested', 'channels'):
                self.assertEqual(after[side]['bcc'][key], before[side]['bcc'][key])

    def test_ranking_retains_pair_decision_and_all_channel_evidence(self):
        from keytao_bot.utils.commonness_query import render_commonness_table
        result = lookup_word_commonness(['敲不死', '情报所'])
        self.assertEqual(result['comparisons'][0]['verdict'], 'not_enough_evidence')
        row = next(row for row in result['words'] if row['word'] == '情报所')
        self.assertEqual(row['bcc']['channels'][0]['count'], 34)
        self.assertEqual(row['corpusFrequency'], 8)
        text = render_commonness_table(result)
        self.assertIn('BCC', text)
        self.assertIn('多领域 34（每百万 0.08）', text)
        self.assertIn('新闻 99（每百万 0.11）', text)
        self.assertNotIn('perMillion', text)
        absent = next(row for row in result['words'] if row['word'] == '敲不死')
        self.assertFalse(absent['known'])
        self.assertIsNone(absent['rank'])

    def test_jieba_is_a_fallback_when_both_words_are_absent_from_bcc(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany('INSERT INTO word_commonness VALUES (?, ?, ?, ?)',
                [('回退甲', 200, 'n', 1), ('回退乙', 20, 'n', 1)])
        result = asyncio.run(review.compare_word_commonness('回退甲', '回退乙'))
        self.assertEqual(result['verdict'], 'front_more_common')
        self.assertEqual(result['decisionReason'], 'frequency_ratio')

    def test_bcc_overrules_conflicting_jieba_and_uses_normalized_channels(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('敲不死', 99999999, 'n', 3)")
            connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('news_total_word_freq.txt', '敲不死', 30, ?)", (30 * 1000000 / 928179976,))
        result = asyncio.run(review.compare_word_commonness('情报所', '敲不死'))
        self.assertEqual(result['verdict'], 'front_more_common')
        self.assertEqual(result['front']['weights'], {'corpus': 0.75, 'dictionary': 0.25})
        self.assertAlmostEqual(result['front']['bcc']['perMillion'], 99 * 1000000 / 928179976)
        self.assertNotIn('99999999', result['summary'])
        self.assertNotIn('jieba', result['summary'])
        self.assertNotIn('未收录 vs 未收录', result['summary'])

    def test_close_low_counts_and_auxiliary_only_attestation_are_signals(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('multi_domain_total_word_freq.txt', '低频对照', 9, ?)", (9 * 1000000 / 403819602,))
        result = asyncio.run(review.compare_word_commonness('一一化', '低频对照'))
        self.assertEqual(result['verdict'], 'close')
        result = asyncio.run(review.compare_word_commonness('龘', '量子薄荷鸽跃器'))
        self.assertEqual(result['verdict'], 'not_enough_evidence')

    def test_dialogue_only_word_has_the_same_headline_signal(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('dialogue_word_freq.txt', '口语独有词', 60, 1.0)")
            connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('multi_domain_total_word_freq.txt', '多领域独有词', 60, 1.0)")
        result = asyncio.run(review.compare_word_commonness('口语独有词', '多领域独有词'))
        self.assertEqual(result['verdict'], 'close')
        self.assertEqual(result['front']['bcc']['perMillion'], 1.0)

    def test_chain_order_and_eviction_use_the_shared_bcc_decision(self):
        result = asyncio.run(review.rank_code_chain_by_commonness([
            {'word': '敲不死', 'code': 'qbsi', 'type': 'Phrase', 'weight': 100},
            {'word': '情报所', 'code': 'qbsi', 'type': 'Phrase', 'weight': 101},
        ], semantic_review_loader=AsyncMock(side_effect=AssertionError('BCC must decide without a model'))))
        self.assertEqual(result['status'], 'ask')
        result = asyncio.run(review.assess_explicit_code_commonness(
            '情报所', 'qbsi', [{'word': '敲不死', 'type': 'Phrase'}]))
        self.assertEqual(result[0]['verdict'], 'not_enough_evidence')

    def test_absent_coinage_can_override_bcc_attested_dictionary_occupant(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('一一化', NULL, 'n', 2)")
        semantic = {'preSubmitAudit': {'semanticContextAutoPassItems': [{
            'word': '沃集鲜', 'assessment': {'accepted': True, 'confidence': .99,
                'meaning': 'fixture modern use', 'nonObscurity': {
                    'route': 'common_characters_and_llm',
                    'characterReferences': [{'corpusFrequency': 10000}] * 3}}}]}}
        comparison = asyncio.run(review.compare_word_commonness('沃集鲜', '一一化'))
        self.assertFalse(comparison['front']['bcc']['attested'])
        self.assertTrue(comparison['behind']['bcc']['attested'])
        self.assertEqual(comparison['verdict'], 'behind_more_common')
        for words in (('沃集鲜', '一一化'), ('一一化', '沃集鲜')):
            loader = AsyncMock(side_effect=lambda word: semantic if word == '沃集鲜' else {})
            result = asyncio.run(review.rank_code_chain_by_commonness([
                {'word': word, 'code': 'wjx', 'type': 'Phrase', 'weight': index}
                for index, word in enumerate(words)], semantic_review_loader=loader))
            self.assertEqual(result['proposedOrder'][0]['word'], '沃集鲜')
            self.assertEqual(result['comparisons'][0]['decisionReason'], 'modern_semantic_vs_dictionary_dominated')
            loader.assert_awaited_once_with('沃集鲜')

# Source: test_s63_bcc_delivery.py
class BccDeliveryTests(unittest.TestCase):
    setUp = BccDeliveryTests_setUp

    def test_registered_read_only_tool_returns_bcc_counts(self):
        manager = SkillsManager()
        manager.load_skill(Path(__file__).parents[1] / 'keytao_bot/skills/keytao-lookup')
        tool = manager.get_tool_function('keytao_word_commonness')
        result = asyncio.run(tool(words=['情报所', '敲不死']))
        self.assertEqual(result['words'][0]['bcc']['channels'][0]['count'], 34)
        self.assertEqual(result['comparisons'][0]['verdict'], 'not_enough_evidence')
        history = {row['channel']: row for row in result['words'][0]['bcc']['historicalChannels']}
        self.assertEqual(history['近代汉语']['count'], 31)
        self.assertIn('古代汉语', history)

    def test_s59_stage_delivers_bcc_copy_without_a_model_or_write(self):
        self.run_stage('词频排序：情报所 敲不死', historical=False)

    def test_s59_stage_delivers_explicit_historical_tiebreak(self):
        seed_historical(self.db, counts={'古例甲': (1234, 1), '古例乙': (12, 50)})
        self.run_stage('词频排序：古例乙 古例甲', historical=True)

    run_stage = BccDeliveryTests_run_stage

# Source: test_s63_bcc_ingest.py
class BccIngestTests(unittest.TestCase):
    def test_all_twelve_datasets_are_ingested_with_historical_labels(self):
        prefixes = {**ingest.CHANNEL_PREFIXES, '古代汉语': 'classical_chinese',
                    '近代汉语': 'modern_chinese'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ingest, 'CHANNEL_PREFIXES', prefixes):
                db, cache, datasets = self.cached_corpus(Path(directory))
            result = ingest.ingest(db, cache, datasets, cached_only=True)
            self.assertEqual(len(result['datasets']), 12)
            self.assertEqual(result['rows'], 72)
            self.assertEqual({(row['channel'], row['token_type']) for row in result['datasets']
                              if row['filename'].startswith(('classical_', 'modern_'))},
                             {('古代汉语', 'char'), ('古代汉语', 'word'),
                              ('近代汉语', 'char'), ('近代汉语', 'word')})

    cached_corpus = BccIngestTests_cached_corpus

    def test_cross_token_types_compare_relative_ranks_not_per_million(self):
        from keytao_bot.utils import keytao_review as review
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            with patch.object(ingest, 'download_file', side_effect=AssertionError('Offline only')):
                result = ingest.ingest(db, cache, datasets, cached_only=True)
            self.assertEqual(result['downloaded'], [])
            with patch.object(review, 'reference_db_path', return_value=db):
                char, word = [review._query_commonness_reference(token) for token in ('甲', '甲词')]
            self.assertGreater(char['bcc']['perMillion'], word['bcc']['perMillion'] * 2)
            self.assertEqual(char['bcc']['rankFraction'], 1 / 2)
            self.assertEqual(word['bcc']['rankFraction'], 1 / 10)
            comparison = review._compare_reference_commonness('甲', '甲词', char, word)
            self.assertEqual(comparison['verdict'], 'behind_more_common')
            self.assertEqual(comparison['decisionReason'], 'bcc_relative_rank_ratio')
            inverse = review._compare_reference_commonness('甲词', '甲', word, char)
            self.assertEqual(inverse['verdict'], 'front_more_common')
            self.assertEqual(inverse['summary'], comparison['summary'])
            self.assertIn('表内排名', comparison['summary'])
            # Old optional snapshots without ranks must not compare unlike units.
            char['bcc']['rankFraction'] = None
            self.assertEqual(review._compare_reference_commonness('甲', '甲词', char, word)['verdict'], 'not_enough_evidence')

    def test_low_disk_refuses_before_download_or_staging_and_keeps_db(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            before = db.read_bytes()
            with (patch.object(ingest.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)),
                  patch.object(ingest, 'download_file') as download,
                  patch.object(ingest, '_build_staging') as staging):
                with self.assertRaisesRegex(ingest.IngestError, 'Insufficient disk headroom'):
                    ingest.ingest(db, cache, datasets, cached_only=True)
            download.assert_not_called()
            staging.assert_not_called()
            self.assertEqual(db.read_bytes(), before)

    def test_wal_guard_preserves_database_before_any_dataset_work(self):
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            before = db.read_bytes()
            Path(str(db) + '-wal').touch()
            with (patch.object(ingest, 'download_file') as download,
                  patch.object(ingest, '_build_staging') as staging):
                with self.assertRaisesRegex(ingest.IngestError, 'has a WAL'):
                    ingest.ingest(db, cache, datasets, cached_only=True)
            download.assert_not_called()
            staging.assert_not_called()
            self.assertEqual(db.read_bytes(), before)

    def test_container_command_starts_bot_after_any_ingest_exit(self):
        root = Path(__file__).resolve().parents[1]
        command = json.loads(next(line[4:] for line in (root / 'Dockerfile').read_text().splitlines() if line.startswith('CMD ')))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            executable = work / 'uv'
            executable.write_text('#!/bin/sh\n'
                'printf "%s\\n" "$*" >> "$BCC_TEST_LOG"\n'
                'case "$*" in\n'
                '  *scripts/ingest_bcc.py*) exit "$BCC_TEST_EXIT" ;;\n'
                'esac\n')
            executable.chmod(0o755)
            for exit_code in (0, 1, 2, 137):
                with self.subTest(exit_code=exit_code):
                    log = work / f'{exit_code}.log'
                    env = {**os.environ, 'PATH': str(work) + os.pathsep + os.environ.get('PATH', ''),
                           'BCC_TEST_LOG': str(log), 'BCC_TEST_EXIT': str(exit_code)}
                    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0)
                    steps = log.read_text().splitlines()
                    self.assertEqual(steps[0], 'run python scripts/build_pinyin_reference.py')
                    self.assertEqual(steps[2], 'run python bot.py')
                    # The per-file timeout is a tunable; pin the shape and a floor that the
                    # largest published table (15.7 MB) can actually finish within.
                    timeout = re.fullmatch(
                        r'run python scripts/ingest_bcc\.py --timeout (\d+)', steps[1])
                    self.assertIsNotNone(timeout, steps[1])
                    self.assertGreaterEqual(int(timeout.group(1)), 60)
                    self.assertEqual('WARNING: BCC ingest failed' in result.stderr, exit_code != 0)

    def test_ingest_cli_reports_outage_or_corrupt_cache_without_publishing(self):
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            inventory = Path(directory) / 'inventory.json'
            inventory.write_text(json.dumps({'datasets': datasets}))
            before = db.read_bytes()
            with (patch.object(sys, 'argv', ['ingest_bcc.py', '--db', str(db)]),
                  patch.object(ingest, 'download_file', side_effect=TimeoutError('fixture outage')),
                  redirect_stderr(io.StringIO()) as error):
                self.assertEqual(ingest.main(), 1)
            self.assertIn('fixture outage', error.getvalue())
            (cache / (datasets[-1]['filename'] + '.zip')).write_bytes(b'broken')
            with (patch.object(sys, 'argv', ['ingest_bcc.py', '--db', str(db), '--cache-dir', str(cache),
                    '--inventory', str(inventory), '--cached-only']),
                  patch.object(ingest, 'download_file', side_effect=AssertionError('Offline only')),
                  redirect_stderr(io.StringIO()) as error):
                self.assertEqual(ingest.main(), 1)
            self.assertIn('not a zip', error.getvalue().lower())
            self.assertEqual(db.read_bytes(), before)

    def test_corrupt_previous_database_is_rebuilt_without_bcc(self):
        from keytao_bot.utils.pinyin_reference_build import build_reference_database
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'example 12 n\n'
            (root / 'jieba.txt').write_bytes(payload)
            (root / 'manifest.json').write_text(json.dumps({'formatVersion': 1, 'datasets': [
                {'id': 'jieba', 'file': 'jieba.txt', 'format': 'jieba', 'import': True,
                 'sha256': hashlib.sha256(payload).hexdigest()}]}))
            db = root / 'reference.db'
            db.write_bytes(b'corrupt old database')
            self.assertTrue(build_reference_database(root, db).rebuilt)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute('PRAGMA quick_check').fetchone()[0], 'ok')
                self.assertEqual(connection.execute('SELECT corpus_frequency FROM word_commonness').fetchone()[0], 12)

    def test_zip_and_csv_integrity_checks_reject_bad_data(self):
        from keytao_bot.utils.bcc_reference import create_schema
        name = 'multi_domain_total_word_freq.txt'
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / 'source.zip'
            for payload, member in ((b'word,count\nword,6\n', name),
                                    (b'token,count\nword,6\nword,6\n', name),
                                    (b'token,count\nword,0\n', name),
                                    (b'token,count\nword,6\n', '../' + name)):
                with self.subTest(payload=payload, member=member):
                    with zipfile.ZipFile(archive_path, 'w') as archive:
                        archive.writestr(member, payload)
                    with closing(sqlite3.connect(':memory:')) as connection:
                        create_schema(connection)
                        with self.assertRaises((ingest.IngestError, sqlite3.IntegrityError)):
                            ingest.load_archive(connection, {'filename': name, 'size_bytes': len(payload)}, archive_path)
            payload = b'token,count\nword,6\n'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr(name, payload)
            data = archive_path.read_bytes().replace(b'word,6', b'word,7')
            archive_path.write_bytes(data)
            with closing(sqlite3.connect(':memory:')) as connection:
                create_schema(connection)
                with self.assertRaises(zipfile.BadZipFile):
                    ingest.load_archive(connection, {'filename': name, 'size_bytes': len(payload)}, archive_path)

    def test_import_is_streamed_normalized_idempotent_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'reference.db'
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY)')
            payloads = {}
            datasets = []
            for channel, prefix in ingest.CHANNEL_PREFIXES.items():
                for kind in ('word', 'char'):
                    name = f'{prefix}_{kind}_freq.txt'
                    text = 'token,count\n甲,3\n乙,1\n'.encode()
                    stream = io.BytesIO()
                    with zipfile.ZipFile(stream, 'w') as archive:
                        archive.writestr(name, text)
                    payloads[name] = stream.getvalue()
                    datasets.append(dict(filename=name, channel=channel, token_type=kind,
                        updated_at='2026-05-22', format='txt', freq_type='freq',
                        token_column='token', count_column='count', size_bytes=len(text)))

            def get(url, destination, timeout):
                name = url.split('/')[-2]
                destination.write_bytes(payloads[name])

            with patch.object(ingest, 'download_file', side_effect=get) as download:
                first = ingest.ingest(db, root / 'cache', datasets, timeout=30)
                self.assertEqual(first['rows'], 24)
                self.assertEqual(download.call_count, 12)
                with closing(sqlite3.connect(db)) as connection:
                    self.assertEqual(connection.execute(
                        "SELECT raw_count, per_million FROM bcc_frequency WHERE token='甲' LIMIT 1"
                    ).fetchone(), (3, 750000.0))
                digest = hashlib.sha256(db.read_bytes()).hexdigest()
                self.assertEqual(ingest.ingest(db, root / 'cache', datasets, timeout=30)['downloaded'], [])
                self.assertEqual(download.call_count, 12)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)
                # A corrupted cached ZIP must be fetched again, never trusted.
                (root / 'cache' / (datasets[0]['filename'] + '.zip')).write_bytes(b'broken')
                ingest.ingest(db, root / 'cache', datasets, timeout=30)
                self.assertEqual(download.call_count, 13)
                digest = hashlib.sha256(db.read_bytes()).hexdigest()
                # A bad later dataset cannot publish an earlier successful update.
                changed = [dict(row, updated_at='2026-06-01') for row in datasets]
                last_good_zip = payloads[datasets[-1]['filename']]
                payloads[datasets[-1]['filename']] = b'not a zip'
                with self.assertRaisesRegex(ingest.IngestError, datasets[-1]['filename']):
                    ingest.ingest(db, root / 'cache', changed, timeout=30)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)
                # A concurrent base-reference replacement must not be overwritten.
                payloads[datasets[-1]['filename']] = last_good_zip

                def concurrent_write(url, destination, timeout):
                    get(url, destination, timeout)
                    db.touch()

                with patch.object(ingest, 'download_file', side_effect=concurrent_write):
                    with self.assertRaisesRegex(ingest.IngestError, 'changed during ingest'):
                        ingest.ingest(db, root / 'cache', changed, timeout=30)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)

    def test_rebuilding_base_reference_preserves_bcc(self):
        from keytao_bot.utils.pinyin_reference_build import build_reference_database
        from keytao_bot.utils.bcc_reference import create_schema
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'example 12 n\n'
            (root / 'jieba.txt').write_bytes(payload)
            (root / 'manifest.json').write_text(json.dumps({'formatVersion': 1, 'datasets': [
                {'id': 'jieba', 'file': 'jieba.txt', 'format': 'jieba', 'import': True,
                 'sha256': hashlib.sha256(payload).hexdigest()}]}))
            db = root / 'reference.db'
            build_reference_database(root, db)
            with closing(sqlite3.connect(db)) as connection, connection:
                create_schema(connection)
                connection.execute("INSERT INTO bcc_dataset VALUES ('x', '多领域', 'word', 'fixture', 'sha', 'zip', 1, 1, 1, 6)")
                connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('x', '情报所', 6, 1000000)")
            (root / 'excluded_words.txt').write_text('example\n')
            self.assertTrue(build_reference_database(root, db).rebuilt)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute('SELECT token, raw_count FROM bcc_frequency').fetchall(), [('情报所', 6)])
                self.assertEqual(connection.execute('SELECT content_sha256 FROM bcc_dataset').fetchone()[0], 'sha')

# Source: test_s64_existing_query.py
class ExistingQueryTests(unittest.TestCase):
    setUp = ExistingQueryTests_setUp

    corpus = ExistingQueryTests_corpus

    def test_existing_query_delivery_with_and_without_bcc_and_pending_candidates(self):
        async def run():
            for mode in CORPUS_MODES:
                with self.corpus(mode):
                    for word, code, phrase_type in (("单份", "dffn", "Phrase"), ("乔布斯", "qbsuv", "Phrase"), ("单", "dfo", "Single")):
                        for pending in ("none", "same_word", "other_word"):
                            with self.subTest(corpus=mode, word=word, pending=pending):
                                fixture = lookup_fixture(word, code, phrase_type)
                                store = harness.MemoryConversationStateStore()
                                key = harness.ConversationAddress.group("qq", "865189947", "s64-fixture")
                                previous = pending_candidate(fixture if pending == "same_word" else lookup_fixture("旧词", "jqci"))
                                if pending != "none":
                                    store.set(key, previous)
                                calls = []

                                async def tool(name, args, *_args, **_kwargs):
                                    self.assertIn(name, fixture, "Unexpected tool or mutation")
                                    calls.append(name)
                                    return json.dumps(fixture[name])

                                with (
                                    patch.object(s64_existing_query_chat, "conversation_state_store", store),
                                    patch.object(s64_existing_query_chat, "call_tool_function", side_effect=tool),
                                    patch.object(s64_existing_query_chat, "_get_simple_word_query_words", AsyncMock(return_value=(word,))),
                                    patch.object(s64_existing_query_chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("No model calls"))),
                                    patch.object(s64_existing_query_chat.logger, "warning") as warnings,
                                ):
                                    raw = await s64_existing_query_commands._try_handle_simple_single_word_query(word, "qq", key.actor_id, key)
                                    self.assertEqual(s64_existing_query_chat._enforce_advertised_reply_contract(raw, key), raw)
                                    ctx = SimpleNamespace(response=raw, conv_key=key, normalized_message_text=word,
                                                          platform="qq", user_id=key.actor_id, simple_word_query_words=(),
                                                          generic_command_intent=harness.MessageCommandIntent())
                                    await s64_existing_query_chat._stage_normalize_response(ctx)
                                    await s64_existing_query_chat._stage_augment_word_query(ctx)
                                    await s64_existing_query_chat._stage_scope_language_only_response(ctx)
                                    await s64_existing_query_chat._stage_append_ticket_challenge(ctx)
                                    before_gate = ctx.response
                                    await s64_existing_query_chat._stage_enforce_advertised_reply_contract(ctx)
                                    self.assertEqual(ctx.response, before_gate)
                                    delivered = s64_existing_query_chat._prepare_user_facing_reply(ctx.response, SimpleNamespace(conversation_address=key, platform="qq"))
                                    self.assertEqual(delivered.splitlines(), [line for line in before_gate.splitlines() if line])
                                    self.assertIn(f"「{word}」已在词库（{code}）", delivered)
                                    self.assertIn("无需操作", delivered)
                                    self.assertIn("追加一个编码", delivered)
                                    self.assertFalse(any("replace_missing_state" in str(call) for call in warnings.call_args_list))
                                    self.assertEqual(calls, ["keytao_lookup_by_word", "keytao_pending_items_by_words", "keytao_lookup_by_words_batch", "keytao_encode"])
                                    if pending == "none":
                                        self.assertEqual(advertised_reply_contract(raw).command_suggestions, ("换码",))
                                        self.assertTrue(s64_existing_query_chat._chat_routing._pending_trusted_word_action_matches(store.get(key), "换码"))
                                        self.assertIsNotNone(parse_explicit_code_request("加入编码 " + code + "o", word))
                                    else:
                                        self.assertEqual(store.get(key), previous)
                                        self.assertNotIn("回复「换码」", delivered)
                                        template = f"给 {word} 加一个码 <code>"
                                        self.assertIn(template, delivered)
                                        parsed = parse_explicit_entry_code_request(template.replace("<code>", code + "o"))
                                        self.assertEqual((parsed.word, parsed.code), (word, code + "o"))
        asyncio.run(run())

    def test_invalid_advertisement_is_still_rejected(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s64-negative")
        store.set(key, pending_candidate(lookup_fixture()))
        unbound = already_existing_word_copy("单份", ("dffn",), can_choose_other_code=True)
        self.assertEqual(advertised_reply_contract(unbound).command_suggestions, ("换码",))
        self.assertEqual(advertised_batch_binding_pairs(unbound), ())
        with patch.object(s64_existing_query_chat, "conversation_state_store", store), patch.object(s64_existing_query_chat.logger, "warning") as warnings:
            self.assertEqual(s64_existing_query_chat._enforce_advertised_reply_contract(unbound, key), "当前没有可验证的可执行操作，本次未写入。")
            self.assertTrue(any("branch=replace_missing_state state=PendingAddWord bindings=0" in str(call) for call in warnings.call_args_list))

    def test_real_orchestrator_builds_and_delivers_existing_candidate_record(self):
        async def run():
            for mode in CORPUS_MODES:
                with self.subTest(corpus=mode), self.corpus(mode):
                    fixture = lookup_fixture()
                    calls = []
                    def tool_call(name, args):
                        return SimpleNamespace(id=name, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                    def tool_function(name):
                        self.assertIn(name, fixture, "Unexpected tool or mutation")
                        async def invoke(**kwargs):
                            calls.append(name)
                            return copy.deepcopy(fixture[name])
                        return invoke
                    model_calls = [tool_call(name, {"words": ["单份"]} if name in {"keytao_pending_items_by_words", "keytao_lookup_by_words_batch"} else {"code": "dffn"} if name == "keytao_lookup_by_code" else {} if name == "keytao_list_draft_items" else {"word": "单份"}) for name in fixture]
                    client = harness._FakeClient([
                        harness._FakeAIResponse("tool_calls", "", model_calls[:3]),
                        harness._FakeAIResponse("tool_calls", "", model_calls[3:]),
                        harness._FakeAIResponse("stop", "「单份」候选编码：\n1. dffn — 已有「单份」\n2. dffno — 空位（推荐）"),
                    ])
                    schemas = [{"type": "function", "function": {"name": name, "description": "Fixture", "parameters": {"type": "object", "properties": {"word": {"type": "string"}, "words": {"type": "array", "items": {"type": "string"}}, "code": {"type": "string"}}}}} for name in fixture]
                    manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)
                    store = harness.MemoryConversationStateStore()
                    context = harness.AgentRequestContext(platform="qq", user_id="s64-fixture", space_type="group", space_id="865189947")
                    agent = harness.AgentOrchestrator(client_factory=lambda: client,
                        runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10),
                        skills_manager=manager, tool_executor=harness.ToolExecutor(tool_function, frozenset()),
                        state_store=store, bind_help_text="fixture", system_prompt_core="fixture")
                    raw = await agent.run("单份", context)
                    record = store.get_record(context.conversation_address)
                    self.assertIsInstance(record.state, harness.PendingAddWord)
                    self.assertEqual(len(client.completions.calls), 3)
                    self.assertEqual(calls, list(fixture))
                    self.assertIn("「单份」已在词库（dffn）", raw)
                    self.assertIn("加入并提交", raw)
                    self.assertIn("回复编号或编码选择", raw)
                    with patch.object(s64_existing_query_chat, "conversation_state_store", store):
                        self.assertEqual(s64_existing_query_chat._enforce_advertised_reply_contract(raw, context.conversation_address), raw)
                        async def repeat_lookup(name, args, *_args, **_kwargs):
                            self.assertIn(name, fixture, "Unexpected tool or mutation")
                            return json.dumps(fixture[name])
                        with patch.object(s64_existing_query_chat, "call_tool_function", side_effect=repeat_lookup), patch.object(
                            s64_existing_query_chat, "_get_simple_word_query_words", AsyncMock(return_value=("单份",)),
                        ):
                            repeated = await s64_existing_query_commands._try_handle_simple_single_word_query(
                                "单份", "qq", context.user_id, context.conversation_address,
                            )
                            self.assertIn("「单份」已在词库（dffn）", repeated)
                            self.assertEqual(s64_existing_query_chat._enforce_advertised_reply_contract(repeated, context.conversation_address), repeated)
                            self.assertEqual(store.get(context.conversation_address), record.state)
        asyncio.run(run())

# Source: test_s65_commonness_render.py
class CommonnessRenderTests(unittest.TestCase):
    setUp = CommonnessRenderTests_setUp

    local_comparison = CommonnessRenderTests_local_comparison

    seed_both = CommonnessRenderTests_seed_both

    assert_delivery = CommonnessRenderTests_assert_delivery

    def test_one_sided_candidate_keeps_bcc_evidence(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        assessment = review._candidate_commonness_assessment({
            "newWord": "敲不死", "occupantWord": "情报所",
            "occupantCode": "qbs", "freeCode": "qbso",
        }, comparison)
        rendered = chat_render._format_candidate_ordering_assessment(
            assessment, {"qbs": 1, "qbso": 2},
        )
        self.assertIn("多领域 34（每百万 0.08）", rendered)
        self.assertIn("新闻 99（每百万 0.11）", rendered)
        self.assertIn("单边", rendered)
        self.assertIn("推荐空位 qbso", rendered)
        self.assertNotIn("常用度信号不足", rendered)
        self.assertNotIn("口语", rendered)
        self.assertNotIn("文学", rendered)
        self.assertNotIn("历史", rendered)
        self.assert_delivery(comparison)

    def test_both_attested_decide_with_both_figures_in_both_directions(self):
        self.seed_both()
        for first, second, verdict in (("蛋粉", "单份", "front_more_common"),
                                       ("单份", "蛋粉", "behind_more_common")):
            with self.subTest(first=first):
                comparison = asyncio.run(review.compare_word_commonness(first, second))
                self.assertEqual(comparison["verdict"], verdict)
                text = self.assert_delivery(comparison, "dffn")
                self.assertIn("每百万 0.09", text)
                self.assertIn("每百万 0.04", text)
                self.assertIn("2.54×", text)
                self.assertIn("「蛋粉」较「单份」更常用", text)
                self.assertNotIn("历史", text)
                self.assertNotIn("新闻", text)

    def test_both_attested_close_does_not_invent_a_winner(self):
        self.seed_both(0.08)
        comparison = asyncio.run(review.compare_word_commonness("蛋粉", "单份"))
        self.assertEqual(comparison["verdict"], "close")
        text = self.assert_delivery(comparison, "dffn")
        self.assertIn("常用度接近", text)
        self.assertNotIn("更常用", text)
        self.assertIn("推荐空位 dffno", text)

    def test_neither_attested_keeps_insufficient_and_historical_supplement(self):
        # Local comparator seam avoids exercising the unrelated web fallback.
        comparison = self.local_comparison("空例甲", "空例乙")
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        text = self.assert_delivery(comparison)
        self.assertIn("常用度信号不足", text)
        self.assertIn("四个现代频道均未收录", text)
        self.assertIn("历史补充", text)
        self.assertIn("未参与判定", text)
        self.assertNotIn("每百万", text)
        self.assertNotIn("古代汉语", text)
        self.assertNotIn("近代汉语", text)

    def test_one_sided_history_is_only_a_supplement(self):
        seed_historical(self.db, counts={"古例甲": (12, 1)})
        comparison = self.local_comparison("古例甲", "古例乙")
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        text = self.assert_delivery(comparison)
        self.assertIn("历史补充", text)
        self.assertIn("古代汉语 12（每百万 1200）", text)
        self.assertIn("未参与判定", text)
        self.assertNotIn("近代汉语", text)

    def test_historical_supplement_is_explicit_only_when_allowed(self):
        seed_historical(self.db, counts={"古例甲": (1234, 1), "古例乙": (12, 50)})
        comparison = asyncio.run(review.compare_word_commonness("古例甲", "古例乙"))
        self.assertEqual(comparison["decisionReason"], "bcc_historical_classical_frequency_ratio")
        text = self.assert_delivery(comparison)
        self.assertIn("现代四频道均未收录", text)
        self.assertIn("按古代汉语频次", text)
        self.assertIn("1,234（每百万 123400）", text)
        self.assertIn("12（每百万 1200）", text)
        self.assertIn("仅作历史语料末级破平", text)
        comparison = asyncio.run(review.compare_word_commonness("古例甲", "情报所"))
        text = self.assert_delivery(comparison)
        self.assertNotIn("古代汉语", text)
        self.assertNotIn("历史补充", text)

    def test_guard_and_explicit_advisory_keep_the_same_comparator_evidence(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        assessment, _, state = candidate_fixture(comparison)
        state.server_candidates = state.server_candidates[:1]
        state.server_ordering_assessments = [{**assessment, "freeCode": ""}]
        guard = pending.candidate_commonness_guard_copy(
            state.word, state.server_candidates, state.server_occupied_words,
            state.server_ordering_assessments,
        )
        protected = harness.chat_commands_module._protected_eviction_response(state, "qbs", ("情报所",))
        advisory = same_code_reorder._advisory("敲不死", "情报所")
        for text in (guard, protected, advisory):
            self.assertIn(comparison["summary"], text)

    def test_render_sites_use_the_shared_evidence_renderer(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        marker = "S65_SHARED_EVIDENCE"
        # Runtime guard: all candidate variants must deliver the shared result.
        for module in (commonness_copy, pending, commonness_query,
                       same_code_reorder, harness.chat_commands_module):
            original = module.render_commonness_summary
            with patch.object(module, "render_commonness_summary", return_value=marker) as render:
                if module is commonness_copy:
                    assessment, _, _ = candidate_fixture(comparison)
                    text = chat_render._format_candidate_ordering_assessment(assessment, {})
                elif module is pending:
                    text = pending.front_insert_recommendation_copy({
                        "newWord": "蛋粉", "occupantWord": "单份", "occupantCode": "dffn",
                        "freeCode": "dffno", "summary": comparison["summary"],
                    })
                elif module is same_code_reorder:
                    text = module._advisory("敲不死", "情报所")
                elif module is commonness_query:
                    from keytao_bot.utils.word_commonness import lookup_word_commonness
                    text = module.render_commonness_table(lookup_word_commonness(["敲不死", "情报所"]))
                else:
                    _, _, state = candidate_fixture(comparison)
                    text = module._protected_eviction_response(state, "qbs", ("情报所",))
                self.assertIn(marker, text, module.__name__)
                render.assert_called()
            self.assertIs(module.render_commonness_summary, original)

# Source: test_s65_commonness_render.py
class CommonnessRenderArchitectureTests(unittest.TestCase):
    # Builders own verdict wording. The other exceptions are parsers/prompts or
    # audit-policy diagnostics, not pairwise user-facing renderers.
    VERDICT_OWNERS = CommonnessRenderArchitectureTests_VERDICT_OWNERS
    NON_RENDERERS = CommonnessRenderArchitectureTests_NON_RENDERERS
    SITES = CommonnessRenderArchitectureTests_SITES

    def test_every_enumerated_render_site_reaches_shared_evidence(self):
        root = Path(review.__file__).parents[1]
        trees = {path: ast.parse((root / path).read_text()) for path in self.SITES}
        trees["utils/commonness_copy.py"] = ast.parse((root / "utils/commonness_copy.py").read_text())
        functions = {node.name: node for tree in trees.values() for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def reaches_shared(name, seen):
            if name == "render_commonness_summary":
                return True
            if name in seen or name not in functions:
                return False
            calls = [node.func.id for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            return any(reaches_shared(call, seen | {name}) for call in calls)

        for path, names in self.SITES.items():
            for name in names:
                with self.subTest(path=path, name=name):
                    self.assertTrue(reaches_shared(name, set()), name)

    def test_no_consumer_hardcodes_a_commonness_verdict(self):
        root = Path(review.__file__).parents[1]
        verdict = re.compile(r"不弱于|更常用|常用度接近|的常用度(?:信号|证据)不足|现有证据未分出高低")
        # Discover new files too: a fourth/fifth renderer must not evade the
        # class guard merely because it was omitted from the site inventory.
        for path in root.rglob("*.py"):
            relative = str(path.relative_to(root))
            tree = ast.parse(path.read_text())
            for function in (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                if (relative, function.name) in self.VERDICT_OWNERS | self.NON_RENDERERS:
                    continue
                for node in ast.walk(function):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertIsNone(verdict.search(node.value),
                                          f"{relative}:{node.lineno} hardcodes a verdict")

# Source: test_s67_ranking.py
class S67RankingTests(unittest.TestCase):
    setUp = S67RankingTests_setUp

    reply = S67RankingTests_reply

    cells = S67RankingTests_cells

    def test_m1_04_commonness_question_without_comparative_is_read_only(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany(
                "INSERT INTO bcc_frequency VALUES ('multi_domain_total_word_freq.txt', ?, ?, ?, ?)",
                [("爆火", 400, 1.0, 1), ("包伙", 100, 0.25, 2)],
            )
        text = asyncio.run(self.reply("爆火和包伙，哪个常用", entries=[]))
        self.assertIn("「爆火」较「包伙」更常用", text)
        self.assertIn("所用频次比 4.00×", text)
        self.assertNotIn("候选编码", text)
        self.assertNotIn("这次服务没有完成处理", text)

    def test_m2_5_frequency_question_compares_both_words_without_add_card(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany(
                "INSERT INTO bcc_frequency VALUES ('multi_domain_total_word_freq.txt', ?, ?, ?, ?)",
                [("枪炮", 400, 1.0, 1), ("抢跑", 100, 0.25, 2)],
            )
        text = asyncio.run(self.reply("枪炮 抢跑，哪个词频更高", entries=[]))
        self.assertIn("「枪炮」较「抢跑」更常用", text)
        self.assertIn("所用频次比 4.00×", text)
        self.assertNotIn("候选编码", text)
        self.assertNotIn("这次服务没有完成处理", text)

    def test_modern_absent_row_names_jieba_and_absence(self):
        text = asyncio.run(self.reply())
        frequency = self.cells(text, "鎗")[2]
        self.assertIn("现代四频道未收录", frequency)
        self.assertIn("jieba 参考 10", frequency)
        self.assertNotRegex(frequency, r"^\d+$")

    def test_current_order_consistent_has_no_advertisement(self):
        for weights in ((10, 11), (10, 10)):
            with self.subTest(weights=weights):
                text = asyncio.run(self.reply(weights=weights))
                self.assertFalse(commonness_query.advertised_command_suggestions(text))

    def test_reversed_current_order_advertises_real_change(self):
        text = asyncio.run(self.reply(weights=(11, 10)))
        self.assertEqual(commonness_query.advertised_command_suggestions(text), (COMMAND,))

    def test_unknown_relation_does_not_use_display_order_as_verdict(self):
        rows = [{"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
                for word, weight in (("嘢", 11), ("鎗", 10))]
        text = asyncio.run(self.reply("词频排序：嘢 鎗", entries=rows))
        self.assertFalse(commonness_query.advertised_command_suggestions(text))

    def test_close_relation_does_not_advertise(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('甲词',100,'n',1)")
            connection.execute("INSERT INTO word_commonness VALUES ('乙词',110,'n',1)")
        rows = [{"word": word, "code": "abcd", "type": "Phrase", "weight": weight}
                for word, weight in (("甲词", 101), ("乙词", 100))]
        text = asyncio.run(self.reply("词频排序：甲词 乙词", entries=rows))
        self.assertFalse(commonness_query.advertised_command_suggestions(text))

    def test_row_status_is_attestation_across_pair_kinds(self):
        for words in (("咽", "嘢"), ("咽", "鎗"), ("之乎", "鎗"), ("敲不死", "空例词")):
            with self.subTest(words=words):
                text = commonness_query.render_commonness_table(lookup_word_commonness(words))
                self.assertIn("词典收录 | 数据收录 | 历史补充", text)
                for word in words:
                    status = self.cells(text, word)[4]
                    if word in ("咽", "嘢"):
                        self.assertEqual(status, "现代 BCC 收录")
                    elif word == "之乎":
                        self.assertEqual(status, "仅历史 BCC 收录")
                    elif word == "鎗":
                        self.assertEqual(status, "jieba、历史 BCC 收录")
                    else:
                        self.assertEqual(status, "无数据")

    def test_second_production_message_is_explicit_about_history(self):
        text = asyncio.run(self.reply(MESSAGES[1]))
        self.assertIn("现代四频道未收录", self.cells(text, "之乎")[2])
        self.assertEqual(self.cells(text, "之乎")[4], "仅历史 BCC 收录")
        self.assertIn("近代汉语 1,311", self.cells(text, "之乎")[5])
        self.assertFalse(commonness_query.advertised_command_suggestions(text))

    def test_verdict_uses_table_precision_two_strongest_channels_and_exact_ratio(self):
        for words in (("曰", "强"), ("强", "曰")):
            result = lookup_word_commonness(words)
            summary = result["comparisons"][0]["summary"]
            self.assertIn("所用频次比 32.09×", summary)
            self.assertLessEqual(sum(channel in summary for channel in CHANNELS), 2)
            self.assertNotIn("975.6388", summary)
            for count, rate in re.findall(r"([\d,]+)（每百万 ([\d.e+-]+)）", summary):
                self.assertIn(f"{count}（每百万 {rate}）", "\n".join(
                    self.cells(commonness_query.render_commonness_table(result), word)[2] for word in words))
            # The headline maximum of each word must remain visible, even in different channels.
            for row in result["words"]:
                strongest = max((r for r in row["bcc"]["channels"] if r["count"] is not None), key=lambda r: r["perMillion"])
                self.assertIn(format_frequency(strongest["count"], strongest["perMillion"]), summary)

    def test_unavailable_bcc_and_other_legacy_sources_never_show_bare_number(self):
        result = lookup_word_commonness(("鎗", "之乎"))
        row = next(row for row in result["words"] if row["word"] == "鎗")
        row["bcc"] = {"available": False}
        for source in ("jieba", "legacy-fixture"):
            row["corpusSource"] = source
            frequency = self.cells(commonness_query.render_commonness_table(result), "鎗")[2]
            self.assertIn(f"{source} 参考 10", frequency)
            self.assertNotIn("未收录", frequency)

    def test_shorter_existing_code_is_not_advertised_as_a_move(self):
        async def run():
            for first_code, second_code, expected in (("abcd", "abcda", ()),
                                                       ("abcda", "abcd", ("把 甲词 放在 乙词 前面",))):
                async def read(name, args):
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [
                            {"word": word, "phrases": [{"word": word, "code": code, "type": "Phrase", "weight": 100}]}
                            for word, code in (("甲词", first_code), ("乙词", second_code))
                        ]}
                    self.assertEqual(name, "keytao_encode")
                    return {"success": True, "word": args["word"], "codes": ["abcd", "abcda"]}
                self.assertEqual(await commonness_query._placement_commands(("甲词", "乙词"), [
                    {"frontWord": "乙词", "behindWord": "甲词", "verdict": "behind_more_common"},
                ], read), expected)
        asyncio.run(run())

    def test_missing_weight_or_duplicate_slot_suppresses_advertisement(self):
        base = [{"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
                for word, weight in (("咽", 11), ("嘢", 10))]
        for rows in ([{k: v for k, v in row.items() if k != "weight"} for row in base],
                     [*base, base[0]]):
            with self.subTest(rows=rows):
                text = asyncio.run(self.reply(entries=rows))
                self.assertFalse(commonness_query.advertised_command_suggestions(text))

    def test_balanced_tiebreak_and_tiny_rate_use_shared_precision(self):
        first, second = [review._query_commonness_reference(word) for word in ("咽", "嘢")]
        # Synthetic equal maxima force the existing balanced-channel tiebreak.
        for reference, balanced in ((first, 0.004123456), (second, 0.00123456)):
            bcc = reference["bcc"]
            bcc["perMillion"] = 10.0
            for row in bcc["channels"]:
                row.update(count=1, perMillion=balanced if row["channel"] == "多领域" else 10.0)
        comparison = review._compare_reference_commonness("咽", "嘢", first, second)
        self.assertIn("balanced_tiebreak", comparison["decisionReason"])
        summary = comparison["summary"]
        self.assertIn("采用多领域比较", summary)
        self.assertIn("所用频次比 3.34×", summary)
        self.assertLessEqual(sum(channel in summary for channel in CHANNELS), 2)
        for value in (0.004123456, 0.00123456):
            self.assertIn(format_frequency(1, value), summary)

    def test_reference_unavailable_is_not_reported_as_modern_absence(self):
        with patch.object(review, "reference_db_path", return_value=self.db.parent / "missing.db"):
            text = commonness_query.render_commonness_table(lookup_word_commonness(("咽", "嘢")))
        for word in ("咽", "嘢"):
            self.assertEqual(self.cells(text, word)[2], "—")
            self.assertEqual(self.cells(text, word)[4], "无数据")
        self.assertNotIn("现代四频道未收录", text)


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            B23CommonnessTests,
            S56CommonnessTests,
            CommonnessBindingTests,
            CommonnessEvidenceTests,
            S59CommonnessRouteTests,
            S62CommonnessParserTests,
            BccCommonnessTests,
            BccDeliveryTests,
            BccIngestTests,
            ExistingQueryTests,
            CommonnessRenderTests,
            CommonnessRenderArchitectureTests,
            S67RankingTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
