from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Content in these tags is markup, navigation or code -- never body text.
# Leaving <script> in means minified JS lands in your index as terms.
_NOISE_TAGS = ("script", "style", "noscript", "template", "svg")


def parse(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Returns (title, text, outbound_links).

    lxml over html.parser: real-world HTML is malformed constantly, and lxml
    recovers from broken nesting the way a browser does rather than giving up.
    """
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # separator=" " matters: without it, "<p>foo</p><p>bar</p>" becomes
    # "foobar" -- one nonsense token instead of two real ones.
    text = soup.get_text(separator=" ", strip=True)

    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        # Resolves "/about" and "../page" against the page's own URL.
        # Without this, every relative link in the corpus is unusable.
        links.append(urljoin(base_url, href))

    return title, text, links
