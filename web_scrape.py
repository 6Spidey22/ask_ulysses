import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from concurrent.futures import ThreadPoolExecutor
import threading

from sentence_transformers import SentenceTransformer
from huggingface_hub import login

import chromadb
import hashlib

from queue import Queue, Empty


#Passing hugging face token to
hf_token = "" #add your own hugging face token here
login(token=hf_token)

#Load a pre-trained model to convert text into vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db") # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_paragraphs")



class MultiThreadedWebCrawler:

    #Initalize
    def __init__(self, start_url):
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc
        self.session = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=10)
        self.visited_lock = threading.Lock()
        self.visited_urls = set()
        self.urls_to_visit = Queue()
        self.urls_to_visit.put(self.start_url)
        self.min_char = 50 #minimum paragraph length
        self.min_words = 5 #minnimum number of words

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
        query_params = parse_qs(fragmentless.query)
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

    def parse(self, html):
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string and "Page Not Found" in soup.title.string:
            return

        #Finds all Links
        anchor_tags = soup.find_all("a", href = True)

        for link in anchor_tags:
            href = link['href']
            if href:
                full_url = self.normalize_url(urljoin(self.start_url, href))
                if urlparse(full_url).netloc == self.domain and full_url not in self.visited_urls:
                    with self.visited_lock:
                        if full_url not in self.visited_urls:
                            self.urls_to_visit.put(full_url)

    #The information scraper
    def scrape_info(self, html):
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string and "Page Not Found" in soup.title.string:
            return

        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text().strip()

            if len(text) < self.min_char:
                continue
            if len(text.split()) < self.min_words:
                continue

            paragraphs.append(text)
        try:
            if paragraphs:
                embeddings = model.encode(paragraphs)
                ids = [self.hash_text(p) for p in paragraphs]  # ← dedupe upgrade

                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=paragraphs,
                    ids=ids
                )
        except Exception as e:
            print(e)

    def post_scrape_callback(self, res):
        try:
            result = res.result()
            if result and result.status_code == 200:
                self.parse(result.text)
                self.scrape_info(result.text)
        finally:
            self.urls_to_visit.task_done()

    def scrape_page(self, url):
        try:
            res = self.session.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return

    def run_web_crawler(self):
        while True:
            try:
                target_url = self.normalize_url(self.urls_to_visit.get(timeout=60))
                if target_url not in self.visited_urls:
                    print("Scraping URL: {}".format(target_url))
                    with self.visited_lock:
                        if target_url in self.visited_urls:
                            return
                        self.visited_urls.add(target_url)
                    job = self.pool.submit(self.scrape_page, target_url)
                    job.add_done_callback(self.post_scrape_callback)
            except Empty:
                if self.urls_to_visit.unfinished_tasks == 0:
                    break
            except Exception as e:
                print(e)
                continue

        self.urls_to_visit.join()
        self.pool.shutdown(wait=True)
        print(f"All URLs visited, number of URLs visited: {len(self.visited_urls)}")
        print(f"URLs to visit: {self.urls_to_visit.qsize}")


if __name__ == "__main__":
    cc = MultiThreadedWebCrawler("https://www.cornellcollege.edu")
    cc.run_web_crawler()
