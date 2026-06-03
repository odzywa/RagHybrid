from dataclasses import dataclass
import html as html_lib
from html.parser import HTMLParser
import json
import re
from typing import List
from urllib.parse import urldefrag, urljoin, urlparse

import requests


SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe", "form"}
BLOCK_TAGS = {
    "article",
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
}
NOISE_CLASS_RE = re.compile(
    r"(nav|navbar|menu|sidebar|toc|breadcrumb|footer|header|cookie|"
    r"advert|banner|promo|search|pagination|feedback|edit-this-page)",
    re.I,
)


@dataclass
class WebPage:
    url: str
    title: str
    text: str
    links: List[str]


class DocumentationHTMLParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts = []
        self.text_parts = []
        self.links = []
        self.skip_depth = 0
        self.in_title = False
        self.in_main = False
        self.main_depth = 0
        self.main_parts = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        attrs_dict = {name.lower(): value or "" for name, value in attrs}

        if self.skip_depth:
            self.skip_depth += 1
            return

        if tag == "a":
            href = attrs_dict.get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))

        if tag == "title":
            self.in_title = True

        if self._is_noise(tag, attrs_dict) or tag in SKIP_TAGS:
            self.skip_depth += 1
            return

        if tag in {"main", "article"} or self._looks_like_main(attrs_dict):
            self.in_main = True
            self.main_depth += 1

        if tag in BLOCK_TAGS:
            self._append("\n", main_only=False)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "title":
            self.in_title = False

        if self.skip_depth:
            self.skip_depth -= 1
            return

        if tag in BLOCK_TAGS:
            self._append("\n", main_only=False)

        if self.in_main and tag in {"main", "article", "div", "section"}:
            self.main_depth = max(0, self.main_depth - 1)
            if self.main_depth == 0:
                self.in_main = False

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
            return

        if self.skip_depth:
            return

        self._append(data, main_only=True)

    def _append(self, data: str, main_only: bool):
        self.text_parts.append(data)
        if self.in_main or not main_only:
            self.main_parts.append(data)

    def _is_noise(self, tag: str, attrs: dict) -> bool:
        if tag in {"nav", "header", "footer", "aside"}:
            return True

        signal = " ".join([
            attrs.get("id", ""),
            attrs.get("class", ""),
            attrs.get("role", ""),
            attrs.get("aria-label", ""),
        ])
        return bool(NOISE_CLASS_RE.search(signal))

    def _looks_like_main(self, attrs: dict) -> bool:
        signal = " ".join([
            attrs.get("id", ""),
            attrs.get("class", ""),
            attrs.get("role", ""),
        ]).lower()
        return any(word in signal for word in ["main", "content", "article", "doc", "markdown"])

    def clean_text(self):
        main_text = normalize_page_text("".join(self.main_parts))
        full_text = normalize_page_text("".join(self.text_parts))

        if len(main_text) >= 300:
            return main_text

        return full_text

    def title(self):
        return normalize_inline_text(" ".join(self.title_parts)) or "Documentation page"


def normalize_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_page_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not is_noise_line(line)]
    return "\n".join(lines).strip()


def is_noise_line(line: str) -> bool:
    lowered = line.lower()
    if len(line) <= 2:
        return True
    if lowered in {"next", "previous", "prev", "edit", "search", "menu", "copy"}:
        return True
    if re.fullmatch(r"[#\-\|/\\\s]+", line):
        return True
    return False


def canonical_url(url: str) -> str:
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(scheme=scheme, netloc=host, path=path).geturl()


def crawl_scope_prefix(start_url: str) -> str:
    path = urlparse(start_url).path or "/"
    parts = [part for part in path.split("/") if part]

    if len(parts) >= 2 and parts[1] == "docs":
        return f"/{parts[0]}/"

    if path.endswith("/"):
        return path
    if "." in path.rsplit("/", 1)[-1]:
        return path.rsplit("/", 1)[0] + "/"
    return path.rstrip("/") + "/"


def extract_next_data(html_text: str, base_url: str):
    match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html_text,
        re.S,
    )

    if not match:
        return None

    try:
        data = json.loads(html_lib.unescape(match.group(1)))
    except Exception:
        return None

    page_props = data.get("props", {}).get("pageProps", {})
    title = (
        page_props.get("metadata", {}).get("title")
        or page_props.get("pageHeading", {}).get("title")
        or data.get("page")
        or base_url
    )
    text_parts = []
    links = []

    metadata = page_props.get("metadata") or {}
    if metadata.get("title"):
        text_parts.append(f"# {metadata['title']}")
    if metadata.get("description"):
        text_parts.append(metadata["description"])

    page_heading = page_props.get("pageHeading") or {}
    if page_heading.get("title") and page_heading.get("title") != metadata.get("title"):
        text_parts.append(f"# {page_heading['title']}")

    page_content = page_props.get("pageContent")
    if page_content:
        text_parts.extend(extract_text_values(page_content))

    mdx_source = page_props.get("mdxSource") or {}
    compiled_source = mdx_source.get("compiledSource", "")
    if compiled_source:
        text_parts.extend(extract_mdx_text(compiled_source))

    links.extend(extract_link_values(page_props, base_url))
    links.extend(extract_mdx_links(compiled_source, base_url))

    return WebPage(
        url=canonical_url(base_url),
        title=normalize_inline_text(title),
        text=normalize_page_text("\n".join(text_parts)),
        links=dedupe_links(links),
    )


def extract_text_values(value):
    parts = []

    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"url", "href", "fullPath", "path", "id", "theme", "iconName", "leadingIconName"}:
                continue
            parts.extend(extract_text_values(item))
    elif isinstance(value, list):
        for item in value:
            parts.extend(extract_text_values(item))
    elif isinstance(value, str):
        clean = normalize_inline_text(value)
        if len(clean) > 2 and not clean.startswith(("/", "#")):
            parts.append(clean)

    return parts


def extract_link_values(value, base_url: str):
    links = []

    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"url", "href", "fullPath"} and isinstance(item, str) and item:
                links.append(urljoin(base_url, item))
            else:
                links.extend(extract_link_values(item, base_url))
    elif isinstance(value, list):
        for item in value:
            links.extend(extract_link_values(item, base_url))

    return links


def extract_mdx_text(compiled_source: str):
    parts = []

    for match in re.finditer(r"`((?:\\`|[^`])*)`", compiled_source):
        value = match.group(1)
        value = value.replace("\\`", "`").replace("\\n", "\n")
        value = re.sub(r"\\u([0-9a-fA-F]{4})", lambda item: chr(int(item.group(1), 16)), value)
        clean = normalize_inline_text(value)
        if len(clean) > 2 and not clean.startswith(("/", "#")):
            parts.append(clean)

    return parts


def extract_mdx_links(compiled_source: str, base_url: str):
    links = []

    for match in re.finditer(r'"(?:href|url|fullPath)"\s*:\s*"([^"]+)"', compiled_source):
        links.append(urljoin(base_url, match.group(1)))

    return links


def dedupe_links(links):
    seen = set()
    clean_links = []

    for link in links:
        url = canonical_url(link)
        if url in seen:
            continue
        seen.add(url)
        clean_links.append(url)

    return clean_links


def is_crawlable_url(url: str, start_host: str, scope_prefix: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc.lower() != start_host.lower():
        return False
    if not parsed.path.startswith(scope_prefix):
        return False
    if re.search(r"\.(png|jpe?g|gif|webp|svg|ico|css|js|zip|tar|gz|pdf|mp4|mp3)$", parsed.path, re.I):
        return False
    return True


def fetch_page(url: str, timeout: int = 15) -> WebPage:
    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "raghybrid-doc-importer/1.0",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower():
        return WebPage(url=url, title=url, text="", links=[])

    response_url = canonical_url(response.url)
    parser = DocumentationHTMLParser(response_url)
    parser.feed(response.text)

    next_page = extract_next_data(response.text, response_url)
    if next_page and len(next_page.text) > len(parser.clean_text()):
        return next_page

    return WebPage(
        url=response_url,
        title=parser.title(),
        text=parser.clean_text(),
        links=dedupe_links(parser.links),
    )


def crawl_documentation(start_url: str, max_pages: int = 50, min_chars: int = 250):
    start = canonical_url(start_url.strip())
    parsed = urlparse(start)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Podaj poprawny URL http/https.")

    max_pages = max(1, min(int(max_pages or 50), 200))
    scope_prefix = crawl_scope_prefix(start)
    pending = [start]
    visited = set()
    pages = []
    skipped = 0
    errors = []

    while pending and len(visited) < max_pages:
        url = pending.pop(0)
        if url in visited:
            continue

        visited.add(url)

        try:
            page = fetch_page(url)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)[:180]})
            continue

        if len(page.text) >= min_chars:
            pages.append(page)
        else:
            skipped += 1

        for link in page.links:
            next_url = canonical_url(link)
            if next_url in visited or next_url in pending:
                continue
            if is_crawlable_url(next_url, parsed.netloc, scope_prefix):
                pending.append(next_url)

    return {
        "start_url": start,
        "scope_prefix": scope_prefix,
        "visited": len(visited),
        "skipped": skipped,
        "errors": errors,
        "pages": pages,
    }
