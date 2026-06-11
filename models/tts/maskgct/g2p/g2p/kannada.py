# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Kannada G2P conversion
# Script: Kannada (Unicode block U+0C80–U+0CFF)
# Phonology reference: Kannada phonology is close to Telugu but has the
# unique phonemes ಳ /ɭ/ and the voiced retroflex fricative ಱ /ɽ/ in
# classical texts.  Modern spoken Kannada merges many Sanskrit aspirates
# with plain stops in native vocabulary.
#
# Output format is identical to tamil.py:
#   • Phoneme tokens separated by |
#   • Word boundaries as |_|
#   • Punctuation tokens wrapped with |

import re

try:
    from indic_numtowords import num2words as indic_num2words
    _INDIC_NUM2WORDS = True
except ImportError:
    _INDIC_NUM2WORDS = False

_NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)*')


def _normalize_numbers_kannada(text: str) -> str:
    """Replace digit sequences with Kannada words (e.g. 5 → ಐದು)."""
    if not _INDIC_NUM2WORDS:
        return text

    def replace_match(m):
        num_str = m.group(0).replace(',', '')
        try:
            return indic_num2words(int(num_str), lang='kn')
        except Exception:
            return m.group(0)

    return _NUMBER_RE.sub(replace_match, text)


# ---------------------------------------------------------------------------
# 1. Vowels (ಸ್ವರಗಳು)
# ---------------------------------------------------------------------------
KANNADA_VOWELS = {
    'ಅ': 'a',  'ಆ': 'aː', 'ಇ': 'i',  'ಈ': 'iː',
    'ಉ': 'u',  'ಊ': 'uː', 'ಋ': 'r̩', 'ಎ': 'e',
    'ಏ': 'eː', 'ಐ': 'aɪ', 'ಒ': 'o',  'ಓ': 'oː',
    'ಔ': 'aʊ',
    'ಂ': 'm',  # anusvara
    'ಃ': 'h',  # visarga
    'ಁ': 'ã',  # chandrabindu
}

# ---------------------------------------------------------------------------
# 2. Vowel Matras (ಗುಣಿತಾಕ್ಷರಗಳು)
# ---------------------------------------------------------------------------
VOWEL_MARKS = {
    'ಾ': 'aː', 'ಿ': 'i',  'ೀ': 'iː', 'ು': 'u',  'ೂ': 'uː',
    'ೃ': 'r̩', 'ೆ': 'e',  'ೇ': 'eː', 'ೈ': 'aɪ', 'ೊ': 'o',
    'ೋ': 'oː', 'ೌ': 'aʊ',
    '್': '',   # Virama — suppresses inherent /a/
}

# ---------------------------------------------------------------------------
# 3. Consonants (ವ್ಯಂಜನಗಳು) — stored with inherent /a/
# ---------------------------------------------------------------------------
KANNADA_CONSONANTS = {
    # Velars
    'ಕ': 'k a',  'ಖ': 'kʰ a', 'ಗ': 'ɡ a',  'ಘ': 'ɡʱ a', 'ಙ': 'ŋ a',
    # Palatals
    'ಚ': 'tʃ a', 'ಛ': 'tʃʰ a','ಜ': 'dʒ a', 'ಝ': 'dʒʱ a','ಞ': 'ɲ a',
    # Retroflexes
    'ಟ': 'ʈ a',  'ಠ': 'ʈʰ a', 'ಡ': 'ɖ a',  'ಢ': 'ɖʱ a', 'ಣ': 'ɳ a',
    # Dentals
    'ತ': 't̪ a', 'ಥ': 't̪ʰ a','ದ': 'd̪ a',  'ಧ': 'd̪ʱ a', 'ನ': 'n a',
    # Labials
    'ಪ': 'p a',  'ಫ': 'pʰ a', 'ಬ': 'b a',  'ಭ': 'bʱ a', 'ಮ': 'm a',
    # Approximants / sibilants
    'ಯ': 'j a',  'ರ': 'r a',  'ಲ': 'l a',  'ವ': 'ʋ a',
    'ಶ': 'ʃ a',  'ಷ': 'ʂ a',  'ಸ': 's a',  'ಹ': 'h a',
    # Kannada-specific
    'ಳ': 'ɭ a',  'ಱ': 'r a',
}

_PUNCT = set(',.?!\':;…')


def _split_punct(word):
    chars = list(word)
    lead = []
    while chars and chars[0] in _PUNCT:
        lead.append(chars.pop(0))
    trail = []
    while chars and chars[-1] in _PUNCT:
        trail.append(chars.pop())
    trail.reverse()
    result = []
    for p in lead:
        result.append((p, True))
    if chars:
        result.append((''.join(chars), False))
    for p in trail:
        result.append((p, True))
    return result


def apply_contextual_rules(ipa_text):
    """
    Kannada sandhi and assimilation rules applied to the unified IPA string.

    Rules:
    1. Gemination (ಆದೇಶ ಸಂಧಿ) — doubling of stops after short vowel
    2. Post-nasal voicing (ಅನುನಾಸಿಕ ಸಂಧಿ)
    3. Intervocalic voicing of voiceless stops (spoken / colloquial Kannada)
    """
    # ------------------------------------------------------------------
    # Rule 1: Gemination
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'k\s+k',   'kː',  ipa_text)
    ipa_text = re.sub(r'ʈ\s+ʈ',   'ʈː',  ipa_text)
    ipa_text = re.sub(r't̪\s+t̪',  't̪ː', ipa_text)
    ipa_text = re.sub(r'p\s+p',   'pː',  ipa_text)
    ipa_text = re.sub(r'tʃ\s+tʃ', 'tʃː', ipa_text)

    # ------------------------------------------------------------------
    # Rule 2: Post-nasal voicing
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'ŋ\s+k',   'ŋ ɡ',  ipa_text)
    ipa_text = re.sub(r'ɲ\s+tʃ',  'ɲ dʒ', ipa_text)
    ipa_text = re.sub(r'ɳ\s+ʈ',   'ɳ ɖ',  ipa_text)
    ipa_text = re.sub(r'n\s+t̪',   'n d̪',  ipa_text)
    ipa_text = re.sub(r'm\s+p',   'm b',  ipa_text)

    # ------------------------------------------------------------------
    # Rule 3: Intervocalic voicing
    # ------------------------------------------------------------------
    vowels = r'(aː|iː|uː|eː|oː|aɪ|aʊ|a|i|u|e|o)'
    ipa_text = re.sub(rf'{vowels}\s+k\s+{vowels}',  r'\1 ɡ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+ʈ\s+{vowels}',  r'\1 ɖ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+t̪\s+{vowels}',  r'\1 d̪ \2', ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+p\s+{vowels}',  r'\1 b \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+tʃ\s+{vowels}', r'\1 dʒ \2', ipa_text)

    return ipa_text


def _word_to_raw_ipa(word_str):
    """Convert a single Kannada word to a space-separated IPA string."""
    ipa_parts = []
    chars = list(word_str)
    i = 0
    while i < len(chars):
        char = chars[i]

        if char in KANNADA_VOWELS:
            ipa_parts.append(KANNADA_VOWELS[char])
            i += 1
            continue

        if char in KANNADA_CONSONANTS:
            base_phone = KANNADA_CONSONANTS[char]
            consonant_part = base_phone.rsplit(' ', 1)[0]

            if i + 1 < len(chars) and chars[i + 1] in VOWEL_MARKS:
                vowel_mark = chars[i + 1]
                if vowel_mark == '್':
                    ipa_parts.append(consonant_part)
                else:
                    ipa_parts.append(consonant_part)
                    ipa_parts.append(VOWEL_MARKS[vowel_mark])
                i += 2
                continue
            else:
                ipa_parts.extend(base_phone.split())
                i += 1
                continue

        ipa_parts.append(char)
        i += 1

    return ' '.join(ipa_parts)


def kannada_to_ipa(text, text_tokenizer=None):
    """
    Convert Kannada text to Amphion IPA token string.

    Output: phonemes separated by |, word boundaries as |_|,
    punctuation wrapped with |.
    """
    if type(text) != str:
        return [kannada_to_ipa(t, text_tokenizer) for t in text]

    text = re.sub(r'\s+', ' ', text).strip()
    text = _normalize_numbers_kannada(text)

    raw_words = text.split(' ')
    token_sequence = []
    for w in raw_words:
        if not w:
            continue
        for tok, is_p in _split_punct(w):
            token_sequence.append((tok, is_p))

    if not token_sequence:
        return ''

    SENTINEL = '\x00'

    ipa_tokens = []
    for tok, is_punct in token_sequence:
        if is_punct:
            ipa_tokens.append(tok)
        else:
            ipa_tokens.append(_word_to_raw_ipa(tok))

    unified = (' ' + SENTINEL + ' ').join(ipa_tokens)
    unified = apply_contextual_rules(unified)

    parts_raw = unified.split(SENTINEL)
    final_parts = []
    for part in parts_raw:
        part = part.strip()
        if not part:
            continue
        phoneme_str = re.sub(r'\s+', '|', part)
        phoneme_str = re.sub(r'\|+', '|', phoneme_str).strip('|')
        if phoneme_str:
            final_parts.append(phoneme_str)

    result = []
    for i, part in enumerate(final_parts):
        result.append(part)
        if i < len(final_parts) - 1:
            next_part = final_parts[i + 1]
            curr_is_punct = len(part) == 1 and part in _PUNCT
            next_is_punct = len(next_part) == 1 and next_part in _PUNCT
            if curr_is_punct or next_is_punct:
                result.append('|')
            else:
                result.append('|_|')

    final_ipa = ''.join(result)
    final_ipa = re.sub(r'\|+', '|', final_ipa)
    final_ipa = final_ipa.strip('|')
    return final_ipa
