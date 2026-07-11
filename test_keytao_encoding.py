import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parent / "keytao_bot" / "utils" / "keytao_encoding.py"
SPEC = importlib.util.spec_from_file_location("keytao_encoding_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
KEYTAO_ENCODING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KEYTAO_ENCODING)
normalize_contextual_phrase_encoding = KEYTAO_ENCODING.normalize_contextual_phrase_encoding
pinyin_to_phonetic_code = KEYTAO_ENCODING.pinyin_to_phonetic_code


def phrase_response(word: str, rate_pinyin: str) -> dict:
    return {
        "input": word,
        "codes": ["fge", "fgeu", "fgeua", "fgeuao"],
        "chars": [
            {"char": "复", "pinyin": "fù", "pinyins": ["fù"], "phoneticCode": "fj", "shapeCode": "uvoi"},
            {"char": "购", "pinyin": "gòu", "pinyins": ["gòu"], "phoneticCode": "gd", "shapeCode": "aoua"},
            {
                "char": "率",
                "pinyin": rate_pinyin,
                "pinyins": ["shuài", "lǜ"],
                "phoneticCode": "eg",
                "shapeCode": "ovaa",
            },
        ],
    }


class ContextualPhraseEncodingTest(unittest.TestCase):
    def test_preserves_umlaut_when_removing_tones(self) -> None:
        self.assertEqual(pinyin_to_phonetic_code("lǜ"), "ll")

    def test_promotes_lv_for_productive_rate_suffix(self) -> None:
        result = normalize_contextual_phrase_encoding("复购率", phrase_response("复购率", "shuài"))

        self.assertEqual(result["chars"][-1]["pinyin"], "lǜ")
        self.assertEqual(result["chars"][-1]["phoneticCode"], "ll")
        self.assertEqual(result["codes"], ["fgl", "fglu", "fglua", "fgluao"])
        self.assertTrue(result["contextPinyinCorrected"])

    def test_keeps_shuai_words_unchanged(self) -> None:
        response = phrase_response("表率", "shuài")
        response["chars"] = response["chars"][-2:]
        response["chars"][0] = {
            "char": "表",
            "pinyin": "biǎo",
            "pinyins": ["biǎo"],
            "phoneticCode": "bc",
            "shapeCode": "vvi",
        }

        result = normalize_contextual_phrase_encoding("表率", response)

        self.assertIs(result, response)
        self.assertEqual(result["chars"][-1]["pinyin"], "shuài")


if __name__ == "__main__":
    unittest.main()
