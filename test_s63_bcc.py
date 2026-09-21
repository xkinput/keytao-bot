"""S63 real-data slice through the shared comparator, tool and ranking route."""
import asyncio
from contextlib import closing
import json
from itertools import product
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.bcc_reference import create_schema
from keytao_bot.utils.word_commonness import lookup_word_commonness

PAIRS = (('情报所', '敲不死'), ('环境法', '户晨风'), ('五角星', '沃集鲜'))
FIXTURE = Path(__file__).parent / 'e2e/fixtures/bcc/s63.json'


def seed_historical(db, channel='古代汉语', counts=None, kind='word', total=10000, rows=100):
    """Synthetic historical evidence, separate from the official corpus slice."""
    prefix = 'classical_chinese' if channel == '古代汉语' else 'modern_chinese'
    dataset = f'{prefix}_{kind}_freq.txt'
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute('DELETE FROM bcc_frequency WHERE dataset=?', (dataset,))
        connection.execute('INSERT OR REPLACE INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (dataset, channel, kind, 'synthetic', 'synthetic', 'synthetic', 1, 1, rows, total))
        for token, (count, rank) in (counts or {}).items():
            connection.execute('INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)',
                               (dataset, token, count, count * 1000000 / total, rank))


class BccCommonnessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = Path(directory.name) / 'reference.db'
        fixture = json.loads(FIXTURE.read_text())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            create_schema(connection)
            connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)')
            for table, rows in (('bcc_dataset', fixture['datasets']), ('bcc_frequency', fixture['rows']), ('word_commonness', fixture['jieba'])):
                for row in rows:
                    connection.execute(f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
        path_patch = patch.object(review, 'reference_db_path', return_value=self.db)
        path_patch.start()
        self.addCleanup(path_patch.stop)
        fallback = patch.object(review, '_estimate_word_commonness_web_fallback', AsyncMock(side_effect=AssertionError('S63 must be local')))
        self.web = fallback.start()
        self.addCleanup(fallback.stop)
        review._clear_review_caches()

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


if __name__ == '__main__':
    unittest.main()
