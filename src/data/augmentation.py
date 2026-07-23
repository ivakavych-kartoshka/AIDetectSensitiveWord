import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT / "resources"


###########################################################

_cache = {}


def load_json(file):

    if file not in _cache:

        with open(
            RESOURCE_DIR / file,
            encoding="utf8",
        ) as f:

            _cache[file] = json.load(f)

    return _cache[file]


###########################################################

HOMOGLYPH = load_json("homoglyph.json")

LEET = load_json("leetspeak.json")

ABBREVIATION = load_json("abbreviation.json")

###########################################################
# Reverse Mapping
###########################################################

INV_HOMO = {}

for fake, real in HOMOGLYPH.items():

    INV_HOMO.setdefault(real, []).append(fake)

###########################################################

INV_LEET = {}

for fake, real in LEET.items():

    INV_LEET.setdefault(real, []).append(fake)

###########################################################


def apply_homoglyph(text):

    for real, fakes in INV_HOMO.items():

        if real in text and random.random() > 0.5:

            text = text.replace(
                real,
                random.choice(fakes),
            )

    return text


###########################################################


def apply_leetspeak(text):

    for real, fakes in INV_LEET.items():

        if real in text and random.random() > 0.5:

            text = text.replace(
                real,
                random.choice(fakes),
            )

    return text


###########################################################


def apply_abbreviation(text):

    keys = list(ABBREVIATION.keys())

    random.shuffle(keys)

    for abbr in keys[: random.randint(1, 4)]:

        original = ABBREVIATION[abbr]

        text = text.replace(original, abbr)

    return text


###########################################################


def random_case(text):

    return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in text)


###########################################################


def add_special(text):

    chars = [
        ".",
        "-",
        "_",
        "*",
        "💀",
        "❤",
        "🔥",
        " ",
    ]

    words = text.split()

    for i in range(len(words)):

        if random.random() > 0.6:

            words[i] += random.choice(chars)

    return " ".join(words)


###########################################################


def add_zero_width(text):

    zw = "\u200b"

    return zw.join(text)


###########################################################


def augment_text(
    text,
    num_variants=10,
):

    techniques = [
        apply_abbreviation,
        apply_homoglyph,
        apply_leetspeak,
        random_case,
        add_special,
        add_zero_width,
    ]

    outputs = []

    for _ in range(num_variants):

        sentence = text

        funcs = random.sample(
            techniques,
            random.randint(2, 5),
        )

        for f in funcs:

            sentence = f(sentence)

        outputs.append(sentence)

    return outputs
