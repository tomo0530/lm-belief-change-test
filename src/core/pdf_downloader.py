import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import PyPDF2
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class PDFDownloader:
    """
    A class to download and extract text from PDFs found on webpages.
    """

    def __init__(self, headless: bool = True, timeout: int = 30, run_dir: Optional[Path] = None):
        """
        Initialize the PDF downloader.

        Args:
            headless: Whether to run browser in headless mode
            timeout: Timeout for web operations in seconds
            run_dir: Directory to save temporary files, if None uses system temp
        """
        self.headless = headless
        self.timeout = timeout
        self.run_dir = run_dir

        # Set up temp directory
        if run_dir:
            self.temp_dir = Path(run_dir) / "tmp"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir = str(self.temp_dir)
        else:
            self.temp_dir = tempfile.mkdtemp()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _setup_driver(self) -> webdriver.Chrome:
        """Setup Chrome WebDriver with appropriate options."""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")

        # Security and performance options
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        # Set download directory
        prefs = {
            "download.default_directory": self.temp_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        try:
            # Use webdriver-manager to automatically handle Chrome driver
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(self.timeout)
            return driver
        except Exception as e:
            self.logger.error(f"Failed to setup Chrome driver: {e}")
            raise

    def _identify_pdf_elements(self, driver: webdriver.Chrome, url: str) -> List[Dict[str, Any]]:
        """
        Identify potential PDF download elements on the page.

        Args:
            driver: Chrome WebDriver instance
            url: The URL being analyzed

        Returns:
            List of dictionaries containing element information
        """
        pdf_elements = []

        # Common selectors for PDF downloads
        pdf_selectors = [
            # Direct PDF links
            "a[href$='.pdf']",
            "a[href*='.pdf']",
            # ArXiv specific
            "a[href*='/pdf/']",
            "a.download-pdf",
            # Generic download buttons/links
            "a[download]",
            "button[download]",
            ".download-button",
            ".pdf-download",
            "a[title*='PDF']",
            "a[title*='Download']",
            "button[title*='PDF']",
            "button[title*='Download']",
            # Text-based identification
            "a:contains('PDF')",
            "a:contains('Download')",
            "button:contains('PDF')",
            "button:contains('Download')",
        ]

        for selector in pdf_selectors:
            try:
                if ":contains(" in selector:
                    # Handle text-based selectors differently
                    text_to_find = selector.split(":contains('")[1].split("')")[0]
                    elements = driver.find_elements(
                        By.XPATH, f"//a[contains(text(), '{text_to_find}')]"
                    )
                    elements.extend(
                        driver.find_elements(
                            By.XPATH, f"//button[contains(text(), '{text_to_find}')]"
                        )
                    )
                else:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)

                for element in elements:
                    try:
                        href = (
                            element.get_attribute("href") or element.get_attribute("onclick") or ""
                        )
                        text = element.text.strip()
                        title = element.get_attribute("title") or ""

                        # Skip if no useful information
                        if not href and not text:
                            continue

                        pdf_elements.append(
                            {
                                "element": element,
                                "href": href,
                                "text": text,
                                "title": title,
                                "selector": selector,
                                "tag": element.tag_name,
                            }
                        )
                    except Exception as e:
                        self.logger.debug(f"Error processing element: {e}")
                        continue

            except Exception as e:
                self.logger.debug(f"Error with selector {selector}: {e}")
                continue

        # Remove duplicates and rank by relevance
        unique_elements = self._deduplicate_and_rank(pdf_elements, url)
        return unique_elements

    def _deduplicate_and_rank(
        self, elements: List[Dict[str, Any]], url: str
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate elements and rank by relevance for PDF download.

        Args:
            elements: List of element dictionaries
            url: The source URL

        Returns:
            Ranked list of unique elements
        """
        seen_hrefs = set()
        unique_elements = []

        for elem in elements:
            href = elem["href"]
            if href and href not in seen_hrefs:
                seen_hrefs.add(href)
                unique_elements.append(elem)
            elif not href and elem["text"]:  # Keep elements with meaningful text even without href
                unique_elements.append(elem)

        # Ranking function
        def rank_element(elem):
            score = 0
            href = elem["href"].lower() if elem["href"] else ""
            text = elem["text"].lower()
            title = elem["title"].lower()

            # High priority indicators
            if ".pdf" in href:
                score += 100
            if "pdf" in text or "pdf" in title:
                score += 50
            if "download" in text or "download" in title:
                score += 30
            if "/pdf/" in href:  # ArXiv style
                score += 80

            # ArXiv specific boost
            if "arxiv.org" in url and "/pdf/" in href:
                score += 200

            # Prefer shorter, cleaner text
            if len(text) < 20 and ("pdf" in text or "download" in text):
                score += 20

            return score

        # Sort by rank (highest first)
        unique_elements.sort(key=rank_element, reverse=True)
        return unique_elements[:5]  # Return top 5 candidates

    def _download_pdf_via_click(
        self, driver: webdriver.Chrome, element_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Download PDF by clicking the identified element.

        Args:
            driver: Chrome WebDriver instance
            element_info: Dictionary containing element information

        Returns:
            Path to downloaded PDF file, or None if failed
        """
        try:
            element = element_info["element"]

            # Scroll element into view
            driver.execute_script("arguments[0].scrollIntoView(true);", element)
            time.sleep(1)

            # Wait for element to be clickable
            wait = WebDriverWait(driver, 10)
            wait.until(EC.element_to_be_clickable(element))

            # Get initial files in download directory
            initial_files = set(os.listdir(self.temp_dir))

            # Click the element
            self.logger.info(f"Clicking element: {element_info['text']} ({element_info['href']})")
            element.click()

            # Wait for download to complete
            download_completed = False
            for _ in range(self.timeout):
                time.sleep(1)
                current_files = set(os.listdir(self.temp_dir))
                new_files = current_files - initial_files

                # Check for completed downloads (no .crdownload files)
                pdf_files = [
                    f for f in new_files if f.endswith(".pdf") and not f.endswith(".crdownload")
                ]
                if pdf_files:
                    download_completed = True
                    pdf_path = os.path.join(self.temp_dir, pdf_files[0])
                    self.logger.info(f"Downloaded PDF: {pdf_path}")
                    return pdf_path

            if not download_completed:
                self.logger.warning("Download timeout - no PDF file detected")
                return None

        except Exception as e:
            self.logger.error(f"Failed to download PDF via click: {e}")
            return None

    def _download_pdf_direct(self, url: str) -> Optional[str]:
        """
        Download PDF directly if URL points to a PDF file.

        Args:
            url: Direct URL to PDF file

        Returns:
            Path to downloaded PDF file, or None if failed
        """
        try:
            self.logger.info(f"Attempting direct PDF download: {url}")
            response = requests.get(url, timeout=self.timeout, stream=True)
            response.raise_for_status()

            # Check if it's actually a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not url.lower().endswith(".pdf"):
                self.logger.warning(f"URL does not seem to be a PDF: {content_type}")
                return None

            # Save to temporary file
            pdf_filename = f"downloaded_{int(time.time())}.pdf"
            pdf_path = os.path.join(self.temp_dir, pdf_filename)

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            self.logger.info(f"Downloaded PDF directly: {pdf_path}")
            return pdf_path

        except Exception as e:
            self.logger.error(f"Failed to download PDF directly: {e}")
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            text_content = []

            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                self.logger.info(f"PDF has {len(pdf_reader.pages)} pages")

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"=== Page {page_num + 1} ===\n{page_text}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue

            full_text = "\n\n".join(text_content)
            self.logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text

        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {e}")
            return ""

    def download_and_extract_pdf(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Main method to download and extract text from PDF on a webpage.

        Args:
            url: URL of the webpage containing PDF link

        Returns:
            Dictionary containing extracted text and metadata, or None if failed
        """
        driver = None
        pdf_path = None

        try:
            # Check if URL directly points to a PDF
            if url.lower().endswith(".pdf") or "/pdf/" in url.lower():
                pdf_path = self._download_pdf_direct(url)
                if pdf_path:
                    text = self._extract_text_from_pdf(pdf_path)
                    return {
                        "text": text,
                        "source_url": url,
                        "pdf_path": pdf_path,
                        "method": "direct_download",
                    }

            # Setup browser driver
            driver = self._setup_driver()

            # Load the webpage
            self.logger.info(f"Loading webpage: {url}")
            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, self.timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Identify PDF download elements
            pdf_elements = self._identify_pdf_elements(driver, url)

            if not pdf_elements:
                self.logger.warning("No PDF download elements found on the page")
                return None

            self.logger.info(f"Found {len(pdf_elements)} potential PDF download elements")

            # Try each element until one works
            for i, element_info in enumerate(pdf_elements):
                self.logger.info(
                    f"Trying element {i + 1}/{len(pdf_elements)}: {element_info['text']}"
                )

                # Try direct download first if href looks like a PDF
                href = element_info["href"]
                if href and (".pdf" in href.lower() or "/pdf/" in href.lower()):
                    full_url = urljoin(url, href)
                    pdf_path = self._download_pdf_direct(full_url)
                    if pdf_path:
                        break

                # Try clicking the element
                pdf_path = self._download_pdf_via_click(driver, element_info)
                if pdf_path:
                    break

            if not pdf_path:
                self.logger.error("Failed to download PDF using any method")
                return None

            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)

            if not text.strip():
                self.logger.warning("No text could be extracted from PDF")
                return None

            return {
                "text": text,
                "source_url": url,
                "pdf_path": pdf_path,
                "method": "selenium_click",
                "num_pages": text.count("=== Page"),
                "char_count": len(text),
            }

        except Exception as e:
            self.logger.error(f"Error in download_and_extract_pdf: {e}")
            return None

        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil

            if self.run_dir:
                # Only clean up files we created in run_dir/tmp
                temp_path = Path(self.temp_dir)
                if temp_path.exists():
                    # Remove all PDF files and other temporary files we created
                    for file_path in temp_path.glob("*"):
                        if file_path.is_file():
                            file_path.unlink()
                    # Remove the tmp directory if it's empty
                    try:
                        temp_path.rmdir()
                        self.logger.info(f"Cleaned up temporary files in {temp_path}")
                    except OSError:
                        # Directory not empty, leave it
                        self.logger.info(
                            f"Cleaned up temporary files in {temp_path} (directory kept)"
                        )
            else:
                # System temp directory - remove completely
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.logger.info("Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temporary files: {e}")
