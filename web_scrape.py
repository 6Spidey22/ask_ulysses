import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, urlencode

from concurrent.futures import ThreadPoolExecutor
import threading

import chromadb
import hashlib

from queue import Queue, Empty

import ollama

import time
import random

#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="vector_db") # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_collection")

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

    def __init__(self, start_url):
        self.stop_event = threading.Event()
        self.domain = urlparse(start_url).netloc
        self.session = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=15)
        self.parse_pool = ThreadPoolExecutor(max_workers=15)
        self.visited_lock = threading.Lock()
        self.visited_urls = set()
        self.urls_to_visit = Queue()
        self.min_words = 3  # minnimum number of words
        self.count = 0
        if ".xml" in start_url:
            response = requests.get(start_url)
            xml_data = response.content
            soup = BeautifulSoup(xml_data, 'xml')
            links = soup.find_all('loc')
            for link in links:
                self.urls_to_visit.put(link.text)
            if home_url not in self.urls_to_visit.queue:
                self.urls_to_visit.put(home_url)
            print(self.urls_to_visit.qsize())
        else:
            self.urls_to_visit.put(start_url)
            if home_url not in self.urls_to_visit.queue:
                self.urls_to_visit.put(home_url)

    def hash_text(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def normalize_url(self, url):
        parsed = urlparse(url)

        # 1. Remove fragment (#section)
        fragmentless = parsed._replace(fragment="")

        # 2. Normalize scheme + domain
        scheme = "https"  # force one scheme (optional but recommended)
        netloc = fragmentless.netloc.lower()

        # 3. Normalize path (remove trailing slash)
        path = fragmentless.path.rstrip("/")

        # 4. Remove tracking query params (optional)
        allowed_params = {}  # or keep some if needed

        query = urlencode(allowed_params, doseq=True)

        normalized = urlunparse((
            scheme,
            netloc,
            path,
            "",
            query,
            ""
        ))

        return normalized

    def parse(self, html, base_url):
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string and "Page Not Found" in soup.title.string:
            return

        #Finds all Links
        anchor_tags = soup.find_all("a", href = True)

        for link in anchor_tags:
            href = link['href']
            if href:
                full_url = self.normalize_url(urljoin(base_url, href))
                if urlparse(full_url).netloc == self.domain and full_url not in self.visited_urls:
                    with self.visited_lock:
                        if full_url not in self.visited_urls:
                            self.urls_to_visit.put(full_url)

    def split_text(self, text, chunk_size = 1200, overlap = 100):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    #The information scraper
    def scrape_info(self, html, url):
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string and "Page Not Found" in soup.title.string:
            return

        self.count += 1
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()

            if len(text.split()) < self.min_words:
                continue

            paragraphs.append(text)

        try:
            if paragraphs:
                all_chunks = []
                for i in paragraphs:
                    chunks = self.split_text(i)
                    all_chunks.extend(chunks)
                embeddings = []
                for chunk in all_chunks:
                    response = ollama.embed(
                        model = 'nomic-embed-text', #all-minilm',
                        input = chunk
                    )
                    embeddings.append(response["embeddings"][0])
                ids = [self.hash_text(c) for c in all_chunks]
                metadatas = [{"source": url} for _ in all_chunks]

                collection.add(
                    embeddings=embeddings,
                    documents=all_chunks,
                    ids=ids,
                    metadatas=metadatas
                )
        except Exception as e:
            print(e)

    def post_scrape_callback(self, res, url):
        try:
            result = res.result()
            if result and result.status_code == 200:
                self.parse(result.text, url)
                self.scrape_info(result.text, url)
        finally:
            self.urls_to_visit.task_done()

    def scrape_page(self, url):
        try:
            session = get_session()
            res = session.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return

    def run_web_crawler(self):
        visited_count = 0
        while True:
            try:
                target_url = self.normalize_url(self.urls_to_visit.get(timeout=60))
                if target_url not in self.visited_urls:
                    visited_count += 1
                    if visited_count % 1000 == 0:
                        time.sleep(random.uniform(5, 15))
                    elif random.uniform(0, 2) <= 0.10:
                        time.sleep(random.uniform(0.5, 1.5))
                    print(f"{self.count + 1} - {visited_count} Scraping URL: {target_url}")
                    with self.visited_lock:
                        if target_url in self.visited_urls:
                            return
                        self.visited_urls.add(target_url)
                    job = self.pool.submit(self.scrape_page, target_url)
                    job.add_done_callback(lambda res, url = target_url: self.parse_pool.submit(self.post_scrape_callback, res, url))
            except Empty:
                if self.urls_to_visit.unfinished_tasks == 0:
                    break
            except Exception as e:
                print(e)
                continue

        self.urls_to_visit.join()
        self.pool.shutdown(wait=True)
        self.parse_pool.shutdown(wait=True)
        print(f"All URLs visited, number of URLs visited: {len(self.visited_urls)}")
        print(f"URLs to visit: {self.urls_to_visit.qsize}")


if __name__ == "__main__":
    cc = MultiThreadedWebCrawler("https://www.cornellcollege.edu/sitemap.xml", "https://www.cornellcollege.edu/")
    cc.run_web_crawler()
