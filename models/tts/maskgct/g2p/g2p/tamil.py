# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
# At the top of tamil.py, after import re:
try:
    from indic_numtowords import num2words as indic_num2words
    _INDIC_NUM2WORDS = True
except ImportError:
    _INDIC_NUM2WORDS = False

_NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)*')

def _normalize_numbers_tamil(text: str) -> str:
    """
    Replace digit sequences in Tamil text with Tamil words.
    Example: "5 கிலோ" → "ஐந்து கிலோ"
             "999 ரூபாய்" → "தொள்ளாயிரத்து தொண்ணூற்றொன்பது ரூபாய்"
    Falls back to leaving digits as-is if indic_numtowords not installed.
    """
    if not _INDIC_NUM2WORDS:
        return text

    def replace_match(m):
        num_str = m.group(0).replace(',', '')  # remove thousand separators
        try:
            num = int(num_str)
            return indic_num2words(num, lang='ta')
        except (ValueError, Exception):
            return m.group(0)  # leave as-is if conversion fails

    return _NUMBER_RE.sub(replace_match, text)

# 1. அடிப்படை உயிரெழுத்துக்கள் (Vowels) மற்றும் ஐ, ஔ (Diphthongs)
TAMIL_VOWELS = {
    'அ': 'a', 'ஆ': 'aː', 'இ': 'i', 'ஈ': 'iː',
    'உ': 'u', 'ஊ': 'uː', 'எ': 'e', 'ஏ': 'eː',
    'ஐ': 'aɪ', 'ஒ': 'o', 'ஓ': 'oː', 'ஔ': 'aʊ'
}

# 2. உயிர்மெய் குறியீடுகள் (Vowel Marks)
VOWEL_MARKS = {
    'ா': 'aː', 'ி': 'i', 'ீ': 'iː', 'ு': 'u', 'ூ': 'uː',
    'ெ': 'e', 'ே': 'eː', 'ை': 'aɪ', 'ொ': 'o', 'ோ': 'oː', 'ௌ': 'aʊ',
    '்': ''  # புள்ளி (Virama) - அகர ஒலியை (inherent schwa) நீக்கும்
}

# 3. அடிப்படை மெய்யெழுத்துக்கள் (Consonants - அகரத்துடன்)
TAMIL_CONSONANTS = {
    'க': 'k a', 'ங': 'ŋ a', 'ச': 's a', 'ஞ': 'ɲ a',
    'ட': 'ʈ a', 'ண': 'ɳ a', 'த': 't̪ a', 'ந': 'n̪ a',
    'ப': 'p a', 'ம': 'm a', 'ய': 'j a', 'ர': 'ɾ a',
    'ல': 'l a', 'வ': 'ʋ a', 'ழ': 'ɻ a', 'ள': 'ɭ a',
    'ற': 'r a', 'ன': 'n a'
}

# 4. கிரந்த மற்றும் சிறப்பு எழுத்துக்கள்
# FIX: க்ஷ removed from here — it cannot be matched by the char-by-char loop
# because it is a 3-char sequence (க + ் + ஷ) and those three chars are
# processed individually as க→k (virama strips 'a'), then ஷ→ʂ a.
# Result is identical: k ʂ a.  The entry would silently never trigger.
GRANTHA_CONSONANTS = {
    'ஜ': 'dʒ a', 'ஷ': 'ʂ a', 'ஸ': 's a', 'ஹ': 'h a',
    'ஃ': 'x'   # ஆய்தம் (āytam) — no inherent vowel
}

# Punctuation marks whose vocab IDs already exist in vocab.json
_PUNCT = set(',.?!\':;…')


def _split_punct(word):
    """
    Yield (token, is_punct) pairs by splitting leading/trailing punctuation
    away from a word token.  This mirrors the pre-processing done by the
    espeak-based languages and by the Chinese cleaner.

    Examples
    --------
    'வணக்கம்!'  →  [('வணக்கம்', False), ('!', True)]
    '"hello"'   →  [('"', True), ('hello', False), ('"', True)]   ← handled
    """
    chars = list(word)
    # collect leading punct
    lead = []
    while chars and chars[0] in _PUNCT:
        lead.append(chars.pop(0))
    # collect trailing punct
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
    தொல்காப்பியம் மற்றும் நன்னூல் அடிப்படையிலான விதிகளை IPA சரத்தில்
    பயன்படுத்துதல்.

    IMPORTANT: this function now receives the FULL sentence IPA string
    (all words joined with spaces, word boundaries NOT yet replaced with |).
    This allows cross-word sandhi rules to fire correctly.

    Longer / more-specific alternations are listed BEFORE shorter ones in
    every regex group to avoid premature partial matches.
    """

    # ------------------------------------------------------------------
    # விதி 4: இரட்டிப்பு வல்லினம் (Gemination of voiceless stops)
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'k\s+k',   'kː',   ipa_text)
    ipa_text = re.sub(r's\s+s',   'tʃː',  ipa_text)  # ச்ச → tʃː
    ipa_text = re.sub(r'ʈ\s+ʈ',   'ʈː',   ipa_text)
    ipa_text = re.sub(r't̪\s+t̪',  't̪ː',  ipa_text)
    ipa_text = re.sub(r'p\s+p',   'pː',   ipa_text)
    ipa_text = re.sub(r'r\s+r',   't r',  ipa_text)  # ற்ற → t r

    # ------------------------------------------------------------------
    # விதி 3: மெல்லினத்தின் பின் வல்லினம் நாதமாதல் (Post-nasal voicing)
    # ------------------------------------------------------------------
    ipa_text = re.sub(r'ŋ\s+k',   'ŋ ɡ',   ipa_text)
    ipa_text = re.sub(r'ɲ\s+s',   'ɲ dʒ',  ipa_text)
    ipa_text = re.sub(r'ɳ\s+ʈ',   'ɳ ɖ',   ipa_text)
    ipa_text = re.sub(r'n̪\s+t̪',  'n̪ d̪',  ipa_text)
    ipa_text = re.sub(r'm\s+p',   'm b',   ipa_text)
    ipa_text = re.sub(r'n\s+r',   'n d r', ipa_text)  # ன்ற → ndr

    # ------------------------------------------------------------------
    # விதி 2: உயிரிடை வல்லினம் நாதமாதல் (Intervocalic voicing)
    # FIX: longer vowel tokens (aː, iː, uː, eː, oː, aɪ, aʊ) MUST precede
    # their shorter prefixes (a, i, u, e, o) in the alternation so that
    # Python's left-to-right alternation does not match the short form and
    # leave the length/diacritic mark dangling.
    # ------------------------------------------------------------------
    vowels = r'(aː|iː|uː|eː|oː|aɪ|aʊ|a|i|u|e|o)'

    ipa_text = re.sub(rf'{vowels}\s+k\s+{vowels}',  r'\1 ɡ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+ʈ\s+{vowels}',  r'\1 ɖ \2',  ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+t̪\s+{vowels}', r'\1 d̪ \2', ipa_text)
    ipa_text = re.sub(rf'{vowels}\s+p\s+{vowels}',  r'\1 b \2',  ipa_text)

    # ------------------------------------------------------------------
    # விதி 8 / Point 13: குற்றியலுகரம் (Word-final short unrounded /u/)
    # FIX: same ordering fix applied to the plosives group — multi-char
    # symbols (tʃ, dʒ, ʈ, ɖ, t̪, d̪) before their single-char prefixes.
    # ------------------------------------------------------------------
    plosives = r'(tʃ|dʒ|ʈ|ɖ|t̪|d̪|k|ɡ|p|b|r)'
    ipa_text = re.sub(rf'{plosives}\s+u\b', r'\1 ʉ', ipa_text)

    return ipa_text


def _word_to_raw_ipa(word_str):
    """
    Convert a single Tamil word string to a space-separated IPA string.
    Returns the IPA tokens joined by spaces, with NO pipe separators yet.
    The returned string has no leading or trailing whitespace.
    """
    ipa_parts = []
    chars = list(word_str)
    i = 0
    while i < len(chars):
        char = chars[i]

        if char in TAMIL_VOWELS:
            ipa_parts.append(TAMIL_VOWELS[char])

        elif char in TAMIL_CONSONANTS or char in GRANTHA_CONSONANTS:
            base_phone = TAMIL_CONSONANTS.get(char, GRANTHA_CONSONANTS.get(char))

            # ஆய்தம் (ஃ) has no inherent vowel — emit the consonant alone
            if char == 'ஃ':
                ipa_parts.append(base_phone)  # 'x'
                i += 1
                continue

            if i + 1 < len(chars) and chars[i + 1] in VOWEL_MARKS:
                vowel_mark = chars[i + 1]
                consonant_part = base_phone.rsplit(' ', 1)[0]  # strip inherent 'a'
                if vowel_mark == '்':
                    # Virama: pure consonant, no vowel
                    ipa_parts.append(consonant_part)
                else:
                    ipa_parts.append(consonant_part)
                    ipa_parts.append(VOWEL_MARKS[vowel_mark])
                i += 1  # consume the vowel mark
            else:
                # Consonant with inherent 'a'
                ipa_parts.extend(base_phone.split())

        else:
            # Unknown / foreign character — pass through as-is
            ipa_parts.append(char)

        i += 1

    return ' '.join(ipa_parts)


def tamil_to_ipa(text, text_tokenizer=None):
    """
    தமிழ் உரையை IPA ஒலியன்களாக மாற்றும் முதன்மைச் சார்பு.

    Output format — identical to the espeak-backed languages (EN/FR/DE/KO):
      • Individual phoneme tokens separated by  |
      • Word boundaries marked with            |_|
      • Punctuation tokens surrounded by        |punct|

    This means phoneme2token() can split on '|' directly and will find
    '_' (vocab ID 4) at every word boundary, exactly as with English.

    PIPELINE
    --------
    1. Normalise whitespace.
    2. Pre-split punctuation away from word tokens.
    3. Convert each word to space-separated IPA tokens.
    4. Join ALL words' IPA with a space-only separator so that
       cross-word sandhi rules (post-nasal voicing, etc.) can fire
       across the boundary.
    5. Apply contextual rules on the unified space-separated string.
    6. Re-split on the internal word-boundary markers that were
       injected in step 3, then assemble the final |_| string.
    """
    if type(text) != str:
        print(f"DEBUG: Tamil batch input: {text}")
        return [tamil_to_ipa(t, text_tokenizer) for t in text]

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\.{2,4}', '…', text)

    text = _normalize_numbers_tamil(text)

    # ------------------------------------------------------------------ #
    # Step 1: tokenise words and pre-split punctuation                    #
    # ------------------------------------------------------------------ #
    raw_words = text.split(' ')

    # Each element is either a Tamil word string or a punctuation char.
    # We keep them in order so we can reconstruct word boundaries later.
    token_sequence = []   # list of (raw_string, is_punct)
    for w in raw_words:
        if not w:
            continue
        for tok, is_p in _split_punct(w):
            token_sequence.append((tok, is_p))

    if not token_sequence:
        return ''

    # ------------------------------------------------------------------ #
    # Step 2: build per-token IPA (still space-separated internally)      #
    # We use a special sentinel  ⟨WORDBND⟩  to mark word boundaries in   #
    # the unified string so that cross-word rules can fire, yet we can    #
    # still locate word boundaries afterward.                             #
    # ------------------------------------------------------------------ #
    SENTINEL = '\x00'   # NULL — never appears in IPA output

    ipa_tokens = []  # list of IPA strings, one per token
    for tok, is_punct in token_sequence:
        if is_punct:
            ipa_tokens.append(tok)          # punctuation kept verbatim
        else:
            ipa_tokens.append(_word_to_raw_ipa(tok))

    # Join the whole sentence with SENTINEL as word-boundary marker so
    # that contextual rules can look across boundaries.
    unified = (' ' + SENTINEL + ' ').join(ipa_tokens)

    # ------------------------------------------------------------------ #
    # Step 3: apply all phonological rules on the full sentence string    #
    # ------------------------------------------------------------------ #
    unified = apply_contextual_rules(unified)

    # ------------------------------------------------------------------ #
    # Step 4: reassemble using Amphion separator conventions              #
    # Split on SENTINEL to recover per-token IPA, then:                  #
    #   • Replace spaces within a token with |                            #
    #   • Join tokens with |_|                                            #
    # ------------------------------------------------------------------ #
    parts_raw = unified.split(SENTINEL)

    final_parts = []
    for part in parts_raw:
        part = part.strip()
        if not part:
            continue
        # Replace intra-token spaces with pipe separators
        phoneme_str = re.sub(r'\s+', '|', part)
        # Collapse any accidental consecutive pipes
        phoneme_str = re.sub(r'\|+', '|', phoneme_str).strip('|')
        if phoneme_str:
            final_parts.append(phoneme_str)

    # Build the final string: phonemes delimited by |, words by |_|
    # Punctuation is treated as a single-token word (like Chinese/English).
    #final_ipa = '|_|'.join(final_parts)

    result = []
    for i, part in enumerate(final_parts):
        result.append(part)
        if i < len(final_parts) - 1:
            next_part = final_parts[i + 1]
            curr_is_punct = len(part) == 1 and part in _PUNCT
            next_is_punct = len(next_part) == 1 and next_part in _PUNCT
            # Punct joins directly with | like English, no _
            if curr_is_punct or next_is_punct:
                result.append('|')
            else:
                result.append('|_|')
    
    final_ipa = ''.join(result)

    # One last safety pass: collapse any double pipes that might arise
    # at the seam between a punctuation token and a word token.
    final_ipa = re.sub(r'\|+', '|', final_ipa)
    # Restore |_| that the above line may have collapsed into |_|
    # (it cannot: '_' is not '|', so |_| always survives re.sub(r'\|+','|',...))
    final_ipa = final_ipa.strip('|')

    #print(f"DEBUG: Cleaner Output (ta): {final_ipa}")
    return final_ipa
