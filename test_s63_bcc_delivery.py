"""S63 fake-tool delivery tests; all corpus facts come from the real slice."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_state_machine as harness
import test_s63_bcc as fixtures
from keytao_bot.skills import SkillsManager


class BccDeliveryTests(unittest.TestCase):
    def setUp(self):
        fixtures.BccCommonnessTests.setUp(self)

    def test_registered_read_only_tool_returns_bcc_counts(self):
        manager = SkillsManager()
        manager.load_skill(Path(__file__).parent / 'keytao_bot/skills/keytao-lookup')
        tool = manager.get_tool_function('keytao_word_commonness')
        result = asyncio.run(tool(words=['情报所', '敲不死']))
        self.assertEqual(result['words'][0]['bcc']['channels'][0]['count'], 34)
        self.assertEqual(result['comparisons'][0]['verdict'], 'not_enough_evidence')

    def test_s59_stage_delivers_bcc_copy_without_a_model_or_write(self):
        chat, commands = harness.openai_chat_module, harness.chat_commands_module
        owner = harness.ConversationAddress.private('qq', 's63-fixture')
        message = '词频排序：情报所 敲不死'
        memory = SimpleNamespace(conversation_address=owner, platform='qq')
        ctx = SimpleNamespace(normalized_message_text=message, platform='qq', user_id=owner.actor_id,
                              conv_key=owner, memory_context=memory, bot=object(), event=object(), QQMessageSegment=None)
        delivered = []

        async def dispatch(name, args, *actor):
            self.assertIn(name, {'keytao_lookup_by_words_batch', 'keytao_encode'})
            return json.dumps({'success': False})

        async def finish(_bot, _event, _user, context, response, _segment):
            delivered.append(chat._prepare_user_facing_reply(response, context))

        async def run():
            token = chat._current_turn_message.set(message)
            try:
                with (patch.object(commands, 'call_tool_function', side_effect=dispatch),
                      patch.object(chat, 'remember_conversation'),
                      patch.object(chat, '_finish_ai_chat_response', side_effect=finish),
                      patch.object(chat, 'get_ai_response_core', side_effect=AssertionError('No model calls')),
                      patch.object(chat, '_classify_message_command_intent', side_effect=AssertionError('No model classifier'))):
                    self.assertTrue(await chat._stage_handle_commonness_query(ctx))
            finally:
                chat._current_turn_message.reset(token)

        asyncio.run(run())
        self.assertEqual(len(delivered), 1)
        self.assertIn('BCC', delivered[0])
        self.assertIn('多领域 34（每百万 0.08）', delivered[0])
        self.assertIn('常用度信号不足', delivered[0])
        self.assertNotIn('「情报所」较「敲不死」更常用', delivered[0])
        for field in ('perMillion', 'raw_count', 'bcc_frequency', 'keytao_word_commonness'):
            self.assertNotIn(field, delivered[0])


if __name__ == '__main__':
    unittest.main()
