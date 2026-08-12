"""Build an in-memory index from a Wikipedia dump and persist it in the
OnDiskIndex format, so a later run of the API (with SEARCH_INDEX_DIR set to
the same directory) can mmap it at startup instead of re-parsing the dump.

Usage:
    python benchmarks/build_index.py --size 20000 --out data/index/20000
"""
import argparse

from search.api.index_loader import DEFAULT_CORPUS_SIZE, DEFAULT_DUMP_PATH, build_in_memory_index
from search.storage.ondisk import OnDiskIndexWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_CORPUS_SIZE,
                         help="number of articles to index")
    parser.add_argument("--dump", default=DEFAULT_DUMP_PATH,
                         help="path to the bz2 MediaWiki XML dump")
    parser.add_argument("--out", required=True,
                         help="output directory for the on-disk index")
    args = parser.parse_args()

    index = build_in_memory_index(args.dump, args.size)

    print(f"writing on-disk index to {args.out}")
    stats = OnDiskIndexWriter(args.out).write(index)
    print(f"wrote {stats['postings_bytes']:,} bytes, "
          f"{stats['terms']:,} terms, {stats['documents']:,} documents")


if __name__ == "__main__":
    main()
