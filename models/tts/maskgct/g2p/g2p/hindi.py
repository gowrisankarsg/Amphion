# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Hindi G2P conversion
# Script: Devanagari (Unicode block U+0900–U+097F)
# Phonology reference: Standard Hindi (Khariboli) has 11 vowels,
# ~35 consonants, 4 aspirated stops, 3 retroflex stops, and the
# critical schwa-deletion rule that distinguishes written from spoken form.
#
# KEY DIFFERENCE from Dravidian languages:
#   Hindi has SCHWA DELETION — the inherent /a/ at the end of a word
#   and before a consonant cluster is often silent. This rule is
#   implemented as a post-processing step after the raw IPA is built.
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


def _normalize_numbers_hindi(text: str) -> str:
    """Replace digit sequences with Hindi words (e.g. 5 → पाँच)."""
    if not _INDIC_NUM2WORDS:
        return text

    def replace_match(m):
        num_str = m.group(0).replace(',', '')
        try:
            return indic_num2words(int(num_str), lang='hi')
        except Exception:
            return m.group(0)

    return _NUMBER_RE.sub(replace_match, text)


# ---------------------------------------------------------------------------
# 1. Vowels (स्वर)
# ---------------------------------------------------------------------------
HINDI_VOWELS = {
    'अ': 'a',  'आ': 'aː', 'इ': 'i',  'ई': 'iː',
    'उ': 'u',  'ऊ': 'uː', 'ऋ': 'r̩', 'ए': 'eː',
    'ऐ': 'aɪ', 'ओ': 'oː', 'औ': 'aʊ',
    'ऑ': 'ɔ',              # borrowed vowel for English words
    'अं': 'ə̃',             # nasalized schwa (rare standalone)
    'ं': 'ã',              # anusvara — nasalizes preceding vowel
    'ँ': 'ã',              # chandrabindu — nasalization
    'ः': 'h',              # visarga
}

# ---------------------------------------------------------------------------
# 2. Vowel Matras (मात्राएँ)
# ---------------------------------------------------------------------------
VOWEL_MARKS = {
    'ा': 'aː', 'ि': 'i',  'ी': 'iː', 'ु': 'u',  'ू': 'uː',
    'ृ': 'r̩', 'े': 'eː', 'ै': 'aɪ', 'ो': 'oː', 'ौ': 'aʊ',
    'ॉ': 'ɔ',              # ऑ matra (borrowed)
    'ं': 'ã',              # anusvara on consonant
    'ँ': 'ã',              # chandrabindu on consonant
    '्': 'VIRAMA',         # Virama — special sentinel handled in logic below
}

# ---------------------------------------------------------------------------
# 3. Consonants (व्यंजन) — stored WITH inherent /a/
#    Hindi (Sanskrit-origin) distinguishes 4-way contrast:
#      voiceless unaspirated / voiceless aspirated /
#      voiced unaspirated    / voiced aspirated
# ---------------------------------------------------------------------------
HINDI_CONSONANTS = {
    # Velars
    'क': 'k a',   'ख': 'kʰ a',  'ग': 'ɡ a',   'घ': 'ɡʱ a',  'ङ': 'ŋ a',
    # Palatals
    'च': 'tʃ a',  'छ': 'tʃʰ a', 'ज': 'dʒ a',  'झ': 'dʒʱ a', 'ञ': 'ɲ a',
    # Retroflexes
    'ट': 'ʈ a',   'ठ': 'ʈʰ a',  'ड': 'ɖ a',   'ढ': 'ɖʱ a',  'ण': 'ɳ a',
    # Dentals
    'त': 't̪ a',  'थ': 't̪ʰ a', 'द': 'd̪ a',   'ध': 'd̪ʱ a', 'न': 'n a',
    # Labials
    'प': 'p a',   'फ': 'pʰ a',  'ब': 'b a',   'भ': 'bʱ a',  'म': 'm a',
    # Approximants
    'य': 'j a',   'र': 'r a',   'ल': 'l a',   'व': 'ʋ a',
    # Sibilants / fricatives
    'श': 'ʃ a',   'ष': 'ʂ a',   'स': 's a',   'ह': 'h a',
    # Nukta consonants (Urdu/Persian/English loanwords)
    'क़': 'q a',  'ख़': 'x a',  'ग़': 'ɣ a',  'ज़': 'z a',
    'ड़': 'ɽ a',  'ढ़': 'ɽʱ a', 'फ़': 'f a',
}

# ---------------------------------------------------------------------------
# 4. Schwa deletion: inherent /a/ is DELETED at end of a word and before
#    a consonant cluster that itself ends with a virama.
#    We implement this as a post-processing pass on the space-joined IPA
#    of each word BEFORE the contextual rules stage.
# ---------------------------------------------------------------------------
def _apply_schwa_deletion(raw_ipa: str) -> str:
    """
    Apply Hindi schwa deletion on the space-separated IPA of a single word.

    Rule (simplified Pandey 1990 / Ohala 1983):
      An inherent /a/ is deleted if:
        (a) it is word-final (last token = 'a'), OR
        (b) it is followed immediately by a consonant cluster
            (two or more IPA consonant tokens in a row without a vowel
             between them) — this happens when a virama caused the NEXT
             consonant to appear bare.

    We detect case (b) by looking at the token sequence:
    if pattern is: ... C a C ... where the second C has no following vowel
    token before another C, delete the 'a'.

    Implementation approach: tokenise on spaces, then walk the list.
    """
    tokens = raw_ipa.split()
    if not tokens:
        return raw_ipa

    # Vowel token set (long forms before short to avoid prefix matching)
    vowels_set = {'aː', 'iː', 'uː', 'eː', 'oː', 'aɪ', 'aʊ', 'a', 'i', 'u', 'e', 'o', 'r̩', 'ã', 'ɔ'}

    # Step A: delete word-final inherent /a/
    if tokens[-1] == 'a':
        tokens = tokens[:-1]

    # Step B: delete /a/ that is sandwiched: C a C [not followed by vowel]
    # i.e. tokens[i] = 'a' and tokens[i-1] is NOT a vowel
    #                       and tokens[i+1] is NOT a vowel
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == 'a':
            prev_is_consonant = (i > 0) and (tokens[i - 1] not in vowels_set)
            next_exists        = (i + 1 < len(tokens))
            next_is_consonant  = next_exists and (tokens[i + 1] not in vowels_set)
            if prev_is_consonant and next_is_consonant:
                i += 1  # delete this /a/
                continue
        result.append(tok)
        i += 1

    return ' '.join(result)


_PUNCT = set(',.?!\':;…।')  # । = Hindi danda (full stop)


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
    Hindi / Devanagari sandhi and assimilation rules on the full IPA string.

    Rules:
    1. Aspirate assimilation across word boundary (rare, kept minimal)
    2. Post-nasal voicing (अनुनासिक सन्धि)
    3. Anusvara place assimilation (before the next consonant)
    """
    # ------------------------------------------------------------------
    # Rule 1: Post-nasal voicing
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'ŋ\s+k',   'ŋ ɡ',  ipa_text)
    ipa_text = re.sub(r'ɲ\s+tʃ',  'ɲ dʒ', ipa_text)
    ipa_text = re.sub(r'ɳ\s+ʈ',   'ɳ ɖ',  ipa_text)
    ipa_text = re.sub(r'n\s+t̪',   'n d̪',  ipa_text)
    ipa_text = re.sub(r'm\s+p',   'm b',  ipa_text)

    # ------------------------------------------------------------------
    # Rule 2: Anusvara place assimilation
    # anusvara /ã/ before a stop takes the stop's place of articulation
    # (simplified: we map ã + velar → ŋ, etc.)
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'ã\s+(k|kʰ|ɡ|ɡʱ)', r'ŋ \1', ipa_text)
    ipa_text = re.sub(r'ã\s+(tʃ|tʃʰ|dʒ|dʒʱ)', r'ɲ \1', ipa_text)
    ipa_text = re.sub(r'ã\s+(ʈ|ʈʰ|ɖ|ɖʱ)', r'ɳ \1', ipa_text)
    ipa_text = re.sub(r'ã\s+(t̪|t̪ʰ|d̪|d̪ʱ)', r'n \1', ipa_text)
    ipa_text = re.sub(r'ã\s+(p|pʰ|b|bʱ)', r'm \1', ipa_text)

    return ipa_text


def _word_to_raw_ipa(word_str):
    """
    Convert a single Hindi (Devanagari) word to a space-separated IPA string,
    then apply schwa deletion.
    """
    ipa_parts = []
    chars = list(word_str)
    i = 0
    while i < len(chars):
        char = chars[i]

        # Standalone vowel letters
        if char in HINDI_VOWELS:
            val = HINDI_VOWELS[char]
            if val:
                ipa_parts.append(val)
            i += 1
            continue

        # Consonants
        if char in HINDI_CONSONANTS:
            base_phone = HINDI_CONSONANTS[char]
            consonant_part = base_phone.rsplit(' ', 1)[0]  # strip inherent 'a'

            if i + 1 < len(chars) and chars[i + 1] in VOWEL_MARKS:
                vowel_mark = chars[i + 1]
                mark_val = VOWEL_MARKS[vowel_mark]

                if mark_val == 'VIRAMA':
                    # Virama: bare consonant, no vowel
                    ipa_parts.append(consonant_part)
                elif mark_val == 'ã':
                    # Nasalisation: consonant + inherent /a/ + nasalisation
                    ipa_parts.append(consonant_part)
                    ipa_parts.append('a')
                    ipa_parts.append('ã')
                else:
                    ipa_parts.append(consonant_part)
                    ipa_parts.append(mark_val)
                i += 2
                continue
            else:
                # Consonant with inherent /a/
                ipa_parts.extend(base_phone.split())
                i += 1
                continue

        # Anusvara / visarga / chandrabindu not attached to a consonant
        if char in VOWEL_MARKS and VOWEL_MARKS[char] not in ('VIRAMA', ''):
            ipa_parts.append(VOWEL_MARKS[char])
            i += 1
            continue

        # Pass through unknowns (digits etc. already normalised)
        ipa_parts.append(char)
        i += 1

    raw_ipa = ' '.join(ipa_parts)
    # Apply schwa deletion per-word
    return _apply_schwa_deletion(raw_ipa)


def hindi_to_ipa(text, text_tokenizer=None):
    """
    Convert Hindi (Devanagari) text to Amphion IPA token string.

    Output: phonemes separated by |, word boundaries as |_|,
    punctuation wrapped with |.

    Key difference from Dravidian G2P:
      Schwa deletion is applied per-word before contextual rules.
    """
    if type(text) != str:
        return [hindi_to_ipa(t, text_tokenizer) for t in text]

    text = re.sub(r'\s+', ' ', text).strip()
    text = _normalize_numbers_hindi(text)

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
