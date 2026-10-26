"""The C++ tokenizer must reproduce Hugging Face `tokenizers` exactly (ids and decode round trip).

Needs a converted model and the original tokenizer.json; skipped when they are absent.
    PYTHONPATH=build-dev python -m pytest tests/test_tokenizer_parity.py -q
"""
import os
import random
from pathlib import Path

import pytest

ternix = pytest.importorskip("ternix_engine")
tokenizers = pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("TERNIX_MODEL", ROOT / "models" / "bitnet-2b4t.ternix"))
TOKJSON = Path(os.environ.get("TERNIX_TOKENIZER_JSON", ROOT / "models" / "bitnet-b1.58-2B-4T-bf16" / "tokenizer.json"))
pytestmark = pytest.mark.skipif(not (MODEL.exists() and TOKJSON.exists()), reason="model files not present")


@pytest.fixture(scope="module")
def pair():
    return ternix.Engine(str(MODEL), gpu=False, threads=1, ctx=64), tokenizers.Tokenizer.from_file(str(TOKJSON))


HAND_WRITTEN = [
    "", " ", "  ", "\n", "\n\n", " \n ", "a", "Hello, world!", "It's they're we've I'm you'll he'd THEY'RE",
    "12345678901234567890", "3.14159 and 2,718", "  leading and trailing  ", "tabs\t\tand\r\nCRLF\r\n\r\n",
    "def f(x):\n    return x**2  # comment\n\n\nclass A:\n    pass\n", "x=1;y=2;z=x+y", "https://example.com/a/b?c=d&e=f#g",
    "Die schnelle braune Fuchs springt \u00fcber den faulen Hund.", "\u4eca\u5929\u5929\u6c14\u5f88\u597d\uff0c\u6211\u4eec\u53bb\u516c\u56ed\u73a9\u5427\u3002",
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645", "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e", "emoji \U0001F600\U0001F680 and \u2764\ufe0f",
    "<|begin_of_text|>hello<|eot_id|>", "User: hi<|eot_id|>Assistant: ", "\u00a0nbsp\u2003em space\u3000ideographic", "a\u0301 combining", "!!!???...---___",
    "( ) [ ] { } < > ; : , . / \\ | ` ~ @ # $ % ^ & * + =", "   \n   \n\n   x", "x   \n", "end.\n\n\n",
]


def test_hand_written_cases(pair):
    eng, hf = pair
    for text in HAND_WRITTEN:
        assert eng.encode(text, True, True) == hf.encode(text, add_special_tokens=True).ids, repr(text)


def test_repository_text(pair):
    eng, hf = pair
    for path in list(ROOT.glob("*.md")) + list((ROOT / "src").glob("*.cpp")) + list((ROOT / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert eng.encode(text, True, True) == hf.encode(text).ids, str(path)


def test_random_unicode(pair):
    eng, hf = pair
    rng = random.Random(7)
    alphabet = [chr(c) for c in list(range(32, 127)) + list(range(0xA0, 0x250)) + list(range(0x400, 0x460)) + list(range(0x4E00, 0x4E60))
                + list(range(0x1F600, 0x1F620))] + ["\n", "\t", "\r", "\u2003", "\u3000"]
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        assert eng.encode(text, True, True) == hf.encode(text).ids, repr(text)


def test_decode_roundtrip(pair):
    eng, _ = pair
    for text in HAND_WRITTEN[3:20]:
        assert eng.decode(eng.encode(text, False, False)).decode("utf-8") == text
