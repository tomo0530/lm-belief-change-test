import re
from pathlib import Path

import fire
import pymupdf
import requests
import yaml
from bs4 import BeautifulSoup
from readability.readability import Document
from rich import print

from src.core.pdf_downloader import PDFDownloader


class WebReader:
    def __init__(self, run_dir=None):
        assert run_dir is not None, "run_dir must be provided"

        self.run_dir = run_dir
        self.downloader = PDFDownloader(headless=True, timeout=30, run_dir=run_dir)

    def extract_webpage_text(self, url_text):
        try:
            # Check if URL is a direct PDF first
            if self._is_direct_pdf_url(url_text) or self._might_contain_pdf(url_text):
                pdf_result = self.downloader.download_and_extract_pdf(url_text)
                if pdf_result and pdf_result.get("text"):
                    print(f"[green]Successfully extracted PDF text from {url_text}[/green]")
                    return dict(
                        text=pdf_result["text"],
                        source_type="pdf",
                        method=pdf_result.get("method", "unknown"),
                        char_count=pdf_result.get("char_count", len(pdf_result["text"])),
                        num_pages=pdf_result.get("num_pages", 0),
                    )

            # Fallback to regular webpage extraction
            resp = requests.get(url_text, timeout=10)
            resp.raise_for_status()

            doc = Document(resp.text)
            main_html = doc.summary()

            soup = BeautifulSoup(main_html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            result = dict(text=text, source_type="webpage")
            return result

        except Exception as e:
            print(f"Error: {e}")
            return None

    def _is_direct_pdf_url(self, url: str) -> bool:
        """
        Check if URL is a direct link to a PDF file.

        Args:
            url: URL to check

        Returns:
            True if URL appears to be a direct PDF link
        """
        url_lower = url.lower()

        # Direct PDF file extensions
        if url_lower.endswith(".pdf"):
            return True

        # PDF files with query parameters
        if ".pdf?" in url_lower or ".pdf#" in url_lower:
            return True

        # Common PDF hosting patterns
        pdf_patterns = [
            r"github.com/.*\.pdf",
            r"raw.githubusercontent.com/.*\.pdf",
            r"/content/.*\.pdf",
            r"/files/.*\.pdf",
            r"/documents/.*\.pdf",
            r"/papers/.*\.pdf",
        ]

        for pattern in pdf_patterns:
            if re.search(pattern, url_lower):
                return True

        return False

    def _might_contain_pdf(self, url: str) -> bool:
        """
        Check if URL might contain a PDF (like ArXiv abstract pages).

        Args:
            url: URL to check

        Returns:
            True if URL might contain a downloadable PDF
        """
        url_lower = url.lower()

        # ArXiv abstract pages
        if "arxiv.org/abs/" in url_lower:
            return True

        # Other academic repositories
        if any(
            domain in url_lower
            for domain in [
                "researchgate.net",
                "academia.edu",
                "semanticscholar.org",
                "ieee.org",
                "acm.org",
                "springer.com",
                "nature.com",
                "science.org",
            ]
        ):
            return True

        return False


def normalize_title(title_text):
    """
    Normalize title text to create safe filenames.

    Args:
        title_text: The title text to normalize

    Returns:
        Normalized title safe for use as filename
    """
    if not title_text or not isinstance(title_text, str):
        return "untitled"

    # Start with the original title
    normalized = title_text.strip()

    # Remove or replace problematic characters for filenames
    # Characters that are problematic on various filesystems: / \ : * ? " < > |
    char_replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "",
        "'": "",
        "<": "_",
        ">": "_",
        "|": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "&": "and",
        "%": "percent",
        "#": "number",
        "@": "at",
        "+": "plus",
        "=": "equals",
        ",": "_",
        ";": "_",
        "!": "",
        "~": "_",
        "`": "",
        "^": "_",
    }

    # Apply character replacements
    for old_char, new_char in char_replacements.items():
        normalized = normalized.replace(old_char, new_char)

    # Replace multiple spaces/underscores with single underscore
    normalized = re.sub(r"[\s_]+", "_", normalized)

    # Remove leading/trailing underscores and dots
    normalized = normalized.strip("_.")

    # Ensure it's not empty after cleaning
    if not normalized:
        return "untitled"

    # Limit length to avoid filesystem issues (most filesystems support 255 chars)
    max_length = 200  # Leave some room for file extensions
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip("_.")

    # Ensure it doesn't start with a dot (hidden file on Unix systems)
    if normalized.startswith("."):
        normalized = "file_" + normalized[1:]

    return normalized


def save_pdf_to_txt(topic_name, title_name):
    OFF_POLICY_CONTENT_DIR = Path("./content/passive_learning")

    doc = pymupdf.open(OFF_POLICY_CONTENT_DIR / f"{topic_name}/{title_name}.pdf")

    texts = []
    for page_num, page in enumerate(doc):  # iterate the document pages
        page_text = page.get_text()  # get plain text encoded as UTF-8
        texts.append(f"=== Page {page_num + 1} ===\n{page_text}")
    print(texts)

    with open(OFF_POLICY_CONTENT_DIR / f"{topic_name}/{title_name}.txt", "w") as f:
        f.write("\n\n".join(texts))


def check(**kwargs):
    OFF_POLICY_CONTENT_DIR = Path("./content/passive_learning")

    name_to_metadata = dict()
    with open("./src/passive_learning/topics.yaml", "r") as f:
        topics = yaml.load(f, Loader=yaml.FullLoader)
    conservative_topics = topics["study"]["conservative"]
    progressive_topics = topics["study"]["progressive"]

    for topic in conservative_topics:
        topic["study_type"] = "conservative"
        name = normalize_title(topic["name"])
        title = normalize_title(topic["title"])
        name_to_metadata[(name, title)] = topic

    for topic in progressive_topics:
        topic["study_type"] = "progressive"
        name = normalize_title(topic["name"])
        title = normalize_title(topic["title"])
        name_to_metadata[(name, title)] = topic

    print("=" * 120)
    for topic_dir in OFF_POLICY_CONTENT_DIR.glob("**/"):
        for file_path in topic_dir.glob("*.txt"):
            with open(file_path, "r") as f:
                content = f.read()
            # print(content[:10000] + "..." if len(content) > 10000 else content)
            name = file_path.parent.name
            title = file_path.name.split(".")[0]

            if (name, title) not in name_to_metadata:
                continue

            study_id = name_to_metadata.get((name, title))["id"]
            study_type = name_to_metadata.get((name, title))["study_type"]
            num_words = len(content.split())
            print(
                f"{name: <20} | {title: <50} | {study_id: <3} | {study_type: <12} | {num_words: <10}"
            )
    print("=" * 120)


def main(**kwargs):
    mode = kwargs.get("mode", "pdf2txt")
    if mode == "pdf2txt":
        topic_name = "Bernie_Sanders"
        title_name = "The_Speech_2010_Bernie_Sanders"
        save_pdf_to_txt(topic_name=topic_name, title_name=title_name)
    elif mode == "check":
        check(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {kwargs.get('mode')}")


if __name__ == "__main__":
    fire.Fire(main)
