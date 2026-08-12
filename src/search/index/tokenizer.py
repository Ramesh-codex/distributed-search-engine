import re
import unicodedata

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "to", "was", "were", "will", "with",
})

# \w+ rather than [a-z0-9]+: the ASCII class silently drops CJK, Cyrillic,
# Arabic and Hebrew entirely -- those documents tokenize to [] and vanish
# from the index with no error. \w is Unicode-aware by default in Python 3.
_TOKEN_RE = re.compile(r"\w+")

# Guards against minified JS, base64 blobs and hex dumps, which match as a
# single 50KB "token" and bloat the vocabulary for zero retrieval value.
_MAX_TOKEN_LEN = 30
def _normalize(text: str) -> str:
    """Lowercase and strip accents so 'Café' and 'cafe' produce the same term."""
    lowered = text.lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))
_SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness",
             "ation", "ement", "ingly", "edly",
             "ing", "ies", "ers", "ed", "es", "er", "ly", "s")

_MIN_STEM = 3


_DOUBLE_EXCEPTIONS = frozenset("lsfz")


def _stem(token: str) -> str:
    """Crude suffix stripper: a deliberate simplification of Porter's algorithm.

    Correctness here means *agreement*, not linguistic accuracy. Every surface
    form of a word must collapse to one term; the term need not be a real word.
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            token = token[: -len(suffix)]
            break

    # 'running' -> 'runn' -> 'run'. Skip ll/ss/ff/zz, which are usually real
    # (caress, will, off, buzz) rather than artifacts of inflection.
    if (
        len(token) > _MIN_STEM
        and token[-1] == token[-2]
        and token[-1] not in _DOUBLE_EXCEPTIONS
    ):
        token = token[:-1]

    # 'cache' -> 'cach' to agree with 'caches'; 'distribute' -> 'distribut'
    # to agree with 'distributed'. Strip the silent e rather than restoring it.
    if len(token) - 1 >= _MIN_STEM and token.endswith("e"):
        token = token[:-1] 

    
    return token


def tokenize(text: str) -> list[str]:
    """Convert raw text into index terms.

    The single entry point for both indexing and querying. One function makes
    analyzer drift between the two paths structurally impossible -- the classic
    silent recall killer in hand-built search.

    Returns a list, not a set: BM25 needs term frequency, so repeats matter.
    """
    normalized = _normalize(text)
    return [
        _stem(token)
        for token in _TOKEN_RE.findall(normalized)
        if token not in STOPWORDS and len(token) <= _MAX_TOKEN_LEN
    ]