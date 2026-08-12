import pytest

from search.index.tokenizer import _stem

VARIANT_GROUPS = [
    ("run", "running"),
    ("cache", "caches"),
    ("distribute", "distributed", "distributing"),
    ("index", "indexing", "indexed"),
    ("shard", "shards", "sharding"),
]


@pytest.mark.parametrize("group", VARIANT_GROUPS)
def test_morphological_variants_share_a_stem(group):
    """A word and its inflections must map to the same term, or recall breaks."""
    stems = {_stem(word) for word in group}
    assert len(stems) == 1, f"{group} produced {stems}"


@pytest.mark.parametrize("word", ["sing", "ties", "bed", "is", "as"])
def test_never_stems_below_min_length(word):
    assert len(_stem(word)) >= min(3, len(word))