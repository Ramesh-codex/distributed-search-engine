import logging
from dataclasses import dataclass

from search.crawler.fetcher import Fetcher
from search.crawler.frontier import Frontier
from search.crawler.parser import parse
from search.index.inverted_index import InvertedIndex

log = logging.getLogger(__name__)


@dataclass
class CrawlStats:
    fetched: int = 0
    indexed: int = 0
    skipped: int = 0
    discovered: int = 0


def crawl(
    seeds: list[str],
    index: InvertedIndex,
    max_pages: int = 100,
    max_depth: int = 3,
) -> CrawlStats:
    """Breadth-first traversal of the web graph, indexing as it goes.

    BFS rather than DFS: depth-first on the web wanders down one site's
    pagination forever. Breadth-first gives broad coverage per unit of time,
    which is what you want when the budget is pages rather than completeness.

    The page budget is the real stop condition -- the frontier grows faster
    than it drains, since one page yields many links, so it never empties.
    """
    frontier = Frontier(seeds, max_depth=max_depth)
    fetcher = Fetcher()
    stats = CrawlStats()
    next_doc_id = 1

    try:
        while stats.indexed < max_pages:
            item = frontier.next()
            if item is None:
                log.info("frontier exhausted after %d pages", stats.indexed)
                break

            url, depth = item
            html = fetcher.fetch(url)
            if html is None:
                stats.skipped += 1
                continue

            stats.fetched += 1
            title, text, links = parse(html, url)

            # A page with almost no text is nav furniture or a redirect stub.
            # Indexing them dilutes avgdl and pollutes the results.
            if len(text) < 200:
                stats.skipped += 1
                continue

            index.add_document(next_doc_id, text, url=url, title=title)
            next_doc_id += 1
            stats.indexed += 1

            for link in links:
                if frontier.add(link, depth + 1):
                    stats.discovered += 1

            log.info("[%d/%d] d=%d %s", stats.indexed, max_pages, depth, title[:60])
    finally:
        fetcher.close()

    return stats
