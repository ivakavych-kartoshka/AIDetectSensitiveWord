import json
import re
import unicodedata

from pathlib import Path

############################################################
# Paths
############################################################

ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = ROOT / "resources"


############################################################
# Cache
############################################################

_CACHE = {}


def load_json(filename):

    if filename not in _CACHE:

        with open(
            RESOURCE_DIR / filename,
            encoding="utf8",
        ) as f:

            _CACHE[filename] = json.load(f)

    return _CACHE[filename]


############################################################
# Resources
############################################################

HOMOGLYPH = load_json("homoglyph.json")

LEETSPEAK = load_json("leetspeak.json")

ABBREVIATION = load_json("abbreviation.json")


############################################################
# Regex
############################################################

EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

URL_PATTERN = re.compile(
    r"(https?://[^\s]+)|(www\.[^\s]+)",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(r"(?:\+84|84|0)(?:\d[\s.-]?){8,10}")

IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

BANK_PATTERN = re.compile(r"\b\d{8,20}\b")

CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")


############################################################
# Unicode
############################################################

ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF\u2028\u2029\u00AD]")


############################################################
# Accent
############################################################


def remove_diacritics(text):

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    return "".join(c for c in text if unicodedata.category(c) != "Mn")


############################################################
# Protect Tokens
############################################################


def protect_tokens(text):

    protected = {}

    counter = 0

    patterns = [
        EMAIL_PATTERN,
        URL_PATTERN,
        PHONE_PATTERN,
        IP_PATTERN,
        CARD_PATTERN,
        BANK_PATTERN,
    ]

    for pattern in patterns:

        while True:

            m = pattern.search(text)

            if not m:

                break

            token = m.group()

            key = f"__TOKEN_{counter}__"

            protected[key] = token

            text = text.replace(
                token,
                key,
                1,
            )

            counter += 1

    return text, protected


############################################################
# Restore Tokens
############################################################


def restore_tokens(
    text,
    protected,
):

    for key, value in protected.items():

        text = text.replace(
            key,
            value,
        )

    return text


############################################################
# Normalize Homoglyph
############################################################


def normalize_homoglyph(text):

    changed = False

    for fake, real in HOMOGLYPH.items():

        if fake in text:

            text = text.replace(fake, real)

            changed = True

    return text, changed


############################################################
# Normalize Leetspeak
############################################################


def normalize_leetspeak(text):

    changed = False

    chars = list(text)

    for i, c in enumerate(chars):

        if c in LEETSPEAK:

            chars[i] = LEETSPEAK[c]

            changed = True

    return "".join(chars), changed


############################################################
# Normalize Abbreviation
############################################################


def normalize_abbreviation(text):

    changed = False

    items = sorted(
        ABBREVIATION.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    for canonical, variants in items:

        if isinstance(variants, str):

            variants = [variants]

        for variant in variants:

            pattern = r"\b" + re.escape(variant.lower()) + r"\b"

            new_text = re.sub(
                pattern,
                canonical.lower(),
                text,
                flags=re.IGNORECASE,
            )

            if new_text != text:

                changed = True

                text = new_text

    return text, changed


############################################################
# Remove Repeated Characters
############################################################

REPEATED_PATTERN = re.compile(
    r"(.)\1{2,}",
    re.IGNORECASE,
)


def normalize_repeated(text):

    changed = False

    while True:

        new_text = REPEATED_PATTERN.sub(
            r"\1",
            text,
        )

        if new_text == text:

            break

        changed = True

        text = new_text

    return text, changed


############################################################
# Clean Spaces
############################################################


def clean_spaces(text):

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


############################################################
# Clean Special Characters
############################################################

SPECIAL_PATTERN = re.compile(r"[^\w\sÀ-ỹ]")


def clean_special(text):

    return SPECIAL_PATTERN.sub(
        " ",
        text,
    )


############################################################
# Normalize Main
############################################################


def normalize_text(text):

    if not isinstance(text, str):

        return {
            "original_normalized": "",
            "no_diacritics": "",
            "combined": "",
            "metadata": {},
        }

    ########################################################

    metadata = {
        "had_unicode": False,
        "had_zero_width": False,
        "had_homoglyph": False,
        "had_leetspeak": False,
        "had_abbreviation": False,
        "had_repeated_chars": False,
    }

    ########################################################
    # Unicode
    ########################################################

    original = text

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    metadata["had_unicode"] = original != text

    ########################################################
    # Lower
    ########################################################

    text = text.lower()

    ########################################################
    # Protect
    ########################################################

    text, protected = protect_tokens(text)

    ########################################################
    # Zero Width
    ########################################################

    new_text = ZERO_WIDTH.sub(
        "",
        text,
    )

    metadata["had_zero_width"] = new_text != text

    text = new_text

    ########################################################
    # Homoglyph
    ########################################################

    text, changed = normalize_homoglyph(text)

    metadata["had_homoglyph"] = changed

    ########################################################
    # Leetspeak
    ########################################################

    text, changed = normalize_leetspeak(text)

    metadata["had_leetspeak"] = changed

    ########################################################
    # Abbreviation
    ########################################################

    text, changed = normalize_abbreviation(text)

    metadata["had_abbreviation"] = changed

    ########################################################
    # Repeated
    ########################################################

    text, changed = normalize_repeated(text)

    metadata["had_repeated_chars"] = changed

    ########################################################
    # Clean
    ########################################################

    text = clean_special(text)

    text = clean_spaces(text)

    ########################################################
    # Restore Protected Tokens
    ########################################################

    text = restore_tokens(
        text,
        protected,
    )

    ########################################################
    # Accent
    ########################################################

    no_diacritics = remove_diacritics(text)

    ########################################################
    # Final
    ########################################################

    combined = clean_spaces(text + " " + no_diacritics)

    return {
        "original_normalized": text,
        "no_diacritics": no_diacritics,
        "combined": combined,
        "metadata": metadata,
    }
