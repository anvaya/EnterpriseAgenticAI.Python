import re
import unicodedata
import base64

# ================================================================
# SECTION 3 — LAYER 0: INPUT NORMALIZATION# 
# ================================================================
LEET_MAP = str.maketrans({
    '0':'o','1':'i','3':'e','4':'a','5':'s',
    '7':'t','@':'a','$':'s','|':'i','!':'i',
})
B64_RE = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')

def normalize_input(text: str, max_length: int = 2000):
    """Layer 0: collapse all character injection attack surfaces."""
    text = unicodedata.normalize('NFKC', text)
    invisible = re.compile(
        r'[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u206a-\u206f\ufeff\u00ad]'
    )
    text = invisible.sub('', text)
    b64_fragments = []
    for m in B64_RE.findall(text):
        try:
            d = base64.b64decode(m + '==').decode('utf-8', errors='ignore')
            if len(d) > 10 and d.isprintable():
                b64_fragments.append(d)
        except Exception:
            pass
    eval_copy = text.translate(LEET_MAP).lower()
    if len(text) > max_length:
        text = text[:max_length] + f'...[TRUNCATED from {len(text)} chars]'
    return text, eval_copy, b64_fragments


