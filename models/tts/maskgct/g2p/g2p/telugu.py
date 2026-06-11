# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Telugu G2P conversion
# Script: Telugu (Unicode block U+0C00–U+0C7F)
# Phonology reference: Telugu has five vowel lengths, distinct retroflexes,
# a nasal syllabic /m/ (anusvara), aspirates borrowed via Sanskrit, and
# the unique syllabic /r̩/ (ఋ).
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


def _normalize_numbers_telugu(text: str) -> str:
    """Replace digit sequences with Telugu words (e.g. 5 → ఐదు)."""
    if not _INDIC_NUM2WORDS:
        return text

    def replace_match(m):
        num_str = m.group(0).replace(',', '')
        try:
            return indic_num2words(int(num_str), lang='te')
        except Exception:
            return m.group(0)

    return _NUMBER_RE.sub(replace_match, text)


# ---------------------------------------------------------------------------
# 1. Vowels (అచ్చులు)
# ---------------------------------------------------------------------------
TELUGU_VOWELS = {
    'అ': 'a',  'ఆ': 'aː', 'ఇ': 'i',  'ఈ': 'iː',
    'ఉ': 'u',  'ఊ': 'uː', 'ఋ': 'r̩', 'ఎ': 'e',
    'ఏ': 'eː', 'ఐ': 'aɪ', 'ఒ': 'o',  'ఓ': 'oː',
    'ఔ': 'aʊ',
    'ం': 'm',  # anusvara
    'ః': 'h',  # visarga
    'ఁ': 'ã',  # chandrabindu
}

# ---------------------------------------------------------------------------
# 2. Vowel Matras (గుణింతాలు)
# ---------------------------------------------------------------------------
VOWEL_MARKS = {
    'ా': 'aː', 'ి': 'i',  'ీ': 'iː', 'ు': 'u',  'ూ': 'uː',
    'ృ': 'r̩', 'ె': 'e',  'ే': 'eː', 'ై': 'aɪ', 'ొ': 'o',
    'ో': 'oː', 'ౌ': 'aʊ',
    '్': '',   # Virama (halant) — suppresses inherent /a/
}

# ---------------------------------------------------------------------------
# 3. Consonants (హల్లులు) — stored with inherent /a/
# ---------------------------------------------------------------------------
TELUGU_CONSONANTS = {
    # Velars
    'క': 'k a',  'ఖ': 'kʰ a', 'గ': 'ɡ a',  'ఘ': 'ɡʱ a', 'ఙ': 'ŋ a',
    # Palatals
    'చ': 'tʃ a', 'ఛ': 'tʃʰ a','జ': 'dʒ a', 'ఝ': 'dʒʱ a','ఞ': 'ɲ a',
    # Retroflexes
    'ట': 'ʈ a',  'ఠ': 'ʈʰ a', 'డ': 'ɖ a',  'ఢ': 'ɖʱ a', 'ణ': 'ɳ a',
    # Dentals
    'త': 't̪ a', 'థ': 't̪ʰ a','ద': 'd̪ a',  'ధ': 'd̪ʱ a', 'న': 'n a',
    # Labials
    'ప': 'p a',  'ఫ': 'pʰ a', 'బ': 'b a',  'భ': 'bʱ a', 'మ': 'm a',
    # Approximants / sibilants
    'య': 'j a',  'ర': 'r a',  'ల': 'l a',  'వ': 'ʋ a',
    'శ': 'ʃ a',  'ష': 'ʂ a',  'స': 's a',  'హ': 'h a',
    # Telugu-specific
    'ళ': 'ɭ a',  'ఱ': 'r a',
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
    Telugu sandhi and assimilation rules on the unified IPA string.

    Rules:
    1. Gemination (ద్విత్వం) after short vowel before stop
    2. Post-nasal voicing (అనుస్వారం తర్వాత)
    3. Intervocalic voicing of voiceless stops (spoken Telugu)
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
    # Rule 3: Intervocalic voicing (spoken / colloquial Telugu)
    # ------------------------------------------------------------------
    vowels = r'(aː|iː|uː|eː|oː|aɪ|aʊ|a|i|u|e|o)'
    ipa_text = re.sub(rf'{vowels}\s+k\s+{vowels}',  r'\1 ɡ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+ʈ\s+{vowels}',  r'\1 ɖ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+t̪\s+{vowels}',  r'\1 d̪ \2', ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+p\s+{vowels}',  r'\1 b \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+tʃ\s+{vowels}', r'\1 dʒ \2', ipa_text)

    return ipa_text


def _word_to_raw_ipa(word_str):
    """Convert a single Telugu word to a space-separated IPA string."""
    ipa_parts = []
    chars = list(word_str)
    i = 0
    while i < len(chars):
        char = chars[i]

        if char in TELUGU_VOWELS:
            ipa_parts.append(TELUGU_VOWELS[char])
            i += 1
            continue

        if char in TELUGU_CONSONANTS:
            base_phone = TELUGU_CONSONANTS[char]
            consonant_part = base_phone.rsplit(' ', 1)[0]

            if i + 1 < len(chars) and chars[i + 1] in VOWEL_MARKS:
                vowel_mark = chars[i + 1]
                if vowel_mark == '్':
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


def telugu_to_ipa(text, text_tokenizer=None):
    """
    Convert Telugu text to Amphion IPA token string.

    Output: phonemes separated by |, word boundaries as |_|,
    punctuation wrapped with |.
    """
    if type(text) != str:
        return [telugu_to_ipa(t, text_tokenizer) for t in text]

    text = re.sub(r'\s+', ' ', text).strip()
    text = _normalize_numbers_telugu(text)

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
