"""Variable-byte codec for delta-encoded postings lists.

A posting is (doc_id, term_frequency). Stored as Python tuples in a list,
each costs roughly 56 bytes of object overhead to carry 8 bytes of data.
Two ideas shrink that:

  Delta encoding. Postings are sorted by doc_id, so store gaps rather than
  absolute ids. [5, 9, 14, 200] becomes [5, 4, 5, 186]. Small numbers.

  Variable-byte encoding. A gap of 4 does not need 8 bytes. VByte spends
  7 bits per byte on data and 1 as a continuation flag, so anything under
  128 fits in one byte. Combined with deltas, most postings become 1-2 bytes.
"""


def encode_varint(value: int) -> bytes:
    """Low 7 bits per byte; high bit set on every byte except the last."""
    if value < 0:
        raise ValueError("varint encoding is unsigned")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)   # continuation
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Returns (value, next_offset)."""
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def encode_postings(postings: list[tuple[int, int]]) -> bytes:
    """Encode a doc_id-sorted postings list to a packed byte string.

    Requires sorted input: gaps are only small if ids ascend, and a negative
    gap cannot be varint-encoded at all. This is where the index's doc_id
    sort order stops being a convention and starts being load-bearing.
    """
    out = bytearray()
    previous = 0
    for doc_id, tf in postings:
        gap = doc_id - previous
        if gap <= 0:
            raise ValueError(f"postings must ascend by doc_id: {doc_id} after {previous}")
        out += encode_varint(gap)
        out += encode_varint(tf)
        previous = doc_id
    return bytes(out)


def decode_postings(data: bytes) -> list[tuple[int, int]]:
    postings = []
    offset = 0
    doc_id = 0
    length = len(data)
    while offset < length:
        gap, offset = decode_varint(data, offset)
        tf, offset = decode_varint(data, offset)
        doc_id += gap
        postings.append((doc_id, tf))
    return postings
