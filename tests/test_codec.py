import pytest
from hypothesis import given, strategies as st

from search.index.codec import (
    decode_postings, decode_varint, encode_postings, encode_varint,
)


@given(st.integers(min_value=0, max_value=2**40))
def test_varint_roundtrip(value):
    assert decode_varint(encode_varint(value), 0)[0] == value


@pytest.mark.parametrize("value,size", [(0, 1), (127, 1), (128, 2), (16383, 2), (16384, 3)])
def test_varint_boundaries(value, size):
    """7 bits per byte, so the size steps at every power of 128."""
    assert len(encode_varint(value)) == size


@given(st.lists(st.tuples(st.integers(1, 10000), st.integers(1, 50)), min_size=1))
def test_postings_roundtrip(raw):
    postings = sorted({doc_id: tf for doc_id, tf in raw}.items())
    assert decode_postings(encode_postings(postings)) == postings


def test_rejects_unsorted():
    with pytest.raises(ValueError):
        encode_postings([(5, 1), (3, 1)])


def test_rejects_duplicate_doc_id():
    with pytest.raises(ValueError):
        encode_postings([(5, 1), (5, 1)])
