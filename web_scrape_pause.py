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

"""
This is an attempt at making a webscraper that is able to pause, with limited time I was unable to fully make it work.
I was aiming to fix the problem with the web scraper that was getting rate limitations after around 2000-3000 URLs
visited in the same domain. The intent was to save the urls already visited and URLs to visit in text files to open and
read on the next run to reset the rate limitations.
"""

# Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="vector_db")  # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_collection")

# creates a Thread-Local Storage
thread_local = threading.local()


# creates a session for the thread to use when requesting a URLs html
# creating a session for a thread specifically is more thread safe than creating a session for all threads combined
def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


# waits for the enter key to be pressed to stop/pause the web scraper
def key_listener(stop_event):
    input("Press ENTER to stop the crawler...\n")
    print("Stopping crawler...")
    stop_event.set()


# Web Crawler Class, this is used to traverse the websight starting at the start url and
# finishes when all urls in the domain have been visited or when stoped
# this pulls all paragraph<p> information from each page and stores it in a local vector database
class MultiThreadedWebCrawler:


    # Initalize
    def __init__(self, start_url):
        self.stop_event = threading.Event()  # creates the stop_event check for stopping the web scraper
        self.domain = urlparse(start_url).netloc
        self.pool = ThreadPoolExecutor(max_workers=15)  # this pool is used for the initial request html retrieval
        self.parse_pool = ThreadPoolExecutor(max_workers=15)  # this pool is used for parsing data
        self.visited_lock = threading.Lock()  # creating a lock on non-thread safe variables
        self.visited_urls = set()  # only unique URLs are stored, no duplicates
        self.urls_to_visit = Queue()  # FIFO
        self.min_words = 3  # minimum number of words
        self.count = 0  # used to watch how many URLs are actually scraped

        if ".xml" in start_url:  # if the starting URL is a .xml(most likely a site map) scrape it first then add all normal urls
            response = requests.get(start_url)
            xml_data = response.content
            soup = BeautifulSoup(xml_data, 'xml')
            links = soup.find_all('loc')
            for link in links:
                self.urls_to_visit.put(link.text)
            print(self.urls_to_visit.qsize())
        elif start_url == "":  # if no url is passed, then the user is trying to start from previously saved state
            print("Restarting crawler from saved point...")
            with open("urls_to_visit.txt", "r") as f:
                for line in f:
                    self.urls_to_visit.put(line.strip())
            with open("visited_urls.txt", "r") as f:
                self.visited_urls = set(line.strip() for line in f)
        else:  # else start at the starting url
            self.urls_to_visit.put(start_url)


    # used to create unique IDs for information in the vector database
    def hash_text(self, text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()


    # normalizes the URL to ensure no identical URLs are being visited even though the string is slightly different
    def normalize_url(self, url):
        parsed = urlparse(url)

        # Removes fragment (#section)
        fragmentless = parsed._replace(fragment="")

        # Normalize scheme
        scheme = "https"  # force one scheme - https
        netloc = fragmentless.netloc.lower()

        # Removes trailing slash
        path = fragmentless.path.rstrip("/")

        allowed_params = {}

        query = urlencode(allowed_params, doseq=True)

        # Final normalized url
        normalized = urlunparse((
            scheme,
            netloc,
            path,
            "",
            query,
            ""
        ))

        return normalized


    # Used to find all URLs<a> in the current URL position and check to see if they have been visited or in the domain
    # if they haven't been visited already and are in the domain, add it to urls_to_visit
    def parse(self, html, base_url):
        if self.stop_event.is_set():
            return
        soup = BeautifulSoup(html, "lxml")

        if soup.title and soup.title.string and "Page Not Found" in soup.title.string:
            return

        # Finds all Links
        anchor_tags = soup.find_all("a", href=True)

        for link in anchor_tags:
            href = link['href']
            if href:
                full_url = self.normalize_url(urljoin(base_url, href))
                if urlparse(full_url).netloc == self.domain and full_url not in self.visited_urls:
                    with self.visited_lock:
                        if full_url not in self.visited_urls:
                            self.urls_to_visit.put(full_url)


    # This is used to chunk data into smaller sizes for embedding. The embedding model can't handle extremely large chunks of information
    # This ensures chunks are small enough, and adds some overlap to the chunks to keep them closer together in the vector databse
    def split_text(self, text, chunk_size=1200, overlap=100):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks


    # The information scraper is used to find all paragraphs<p> and stores them in the vector database next to the URL they were found in
    def scrape_info(self, html, url):
        if self.stop_event.is_set():
            return
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
                        model='nomic-embed-text',  # all-minilm',
                        input=chunk
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


    # After the request has been acquired, the html is passed to parse and scrape_info
    def post_scrape_callback(self, res, url):
        try:
            result = res.result()
            if result and result.status_code == 200:
                self.parse(result.text, url)
                self.scrape_info(result.text, url)
        finally:
            self.urls_to_visit.task_done()


    # Used to request access to a URLs information
    def scrape_page(self, url):
        try:
            session = get_session()
            res = session.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return


    # Saves the state that the web scraper is at for later use
    def pause_scrape(self):
        with open("urls_to_visit.txt", "w") as f:
            f.write("\n".join(list(self.urls_to_visit.queue)))
        with open("visited_urls.txt", "w") as f:
            f.writelines(item + "\n" for item in self.visited_urls)


    # The main portion of the web crawler, used to go through the Queue of unvisited URLs and continues until all URLs are visited or a stop_event(enter is pressed)
    def run_web_crawler(self):
        visited_count = 0
        done = False
        while not self.stop_event.is_set():
            try:
                try:
                    target_url = self.normalize_url(self.urls_to_visit.get(timeout=10))
                except Empty:
                    if self.stop_event.is_set():
                        break
                    continue
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
                    if self.stop_event.is_set():
                        break
                    job = self.pool.submit(self.scrape_page, target_url)
                    job.add_done_callback(lambda res, url=target_url: self.parse_pool.submit(self.post_scrape_callback, res, url))
            except Empty:
                if self.urls_to_visit.unfinished_tasks == 0:
                    print(f"All URLs visited, number of URLs visited: {len(self.visited_urls)}")
                    done = True
                    break
            except Exception as e:
                print(e)
                continue

        # saves the state if not through the Queue
        print("Shutting down thread pools...")
        self.urls_to_visit.join()
        print("pool shut down")
        self.pool.shutdown(wait=True)
        print("pars pool shut down")
        self.parse_pool.shutdown(wait=True)
        if not done:
            print("Saving progress...")
            self.pause_scrape()  # save progress
        print("Crawler stopped.")




if __name__ == "__main__":
    cc = MultiThreadedWebCrawler("https://www.cornellcollege.edu/")

    listener = threading.Thread(target=key_listener, args=(cc.stop_event,), daemon=True)
    listener.start()

    cc.run_web_crawler()

    listener.join()
    print("Done")
