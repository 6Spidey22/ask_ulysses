import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
from huggingface_hub import login


#Global Variables
start_url = "https://www.cornellcollege.edu"
domain = urlparse(start_url).netloc
visited_urls = set()
urls_to_visit = [start_url]

#Passing hugging face token to
hf_token = "" #Put your hf_token here
login(token=hf_token)

#Load a pre-trained model to convert text into vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db") # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_paragraphs")

def web_crawl(url):
    if url in visited_urls:
        #If the url has already been visited then no need to search it again
        return

    print(f"Crawling {url}")
    #Adding current url to visited_urls
    visited_urls.add(url)

    #gets the HTML from the url
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
    else:
        print(f"Error fetching {url}")
        return

    #Finds all paragraphs <p>
    paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]

    if len(paragraphs) != 0:

        #Converts text into vectors
        embeddings = model.encode(paragraphs)

        #Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(paragraphs))]

        #Adds to the vector database
        collection.add(
            embeddings=embeddings.tolist(),
            documents=paragraphs,
            ids=ids
        )

    #Finds all links in the current url and adds them to the urls_to_visit if it's within the domain
    for link in soup.find_all("a"):
        href = link.get("href")
        if href:
            full_url = urljoin(url, href)
            if urlparse(full_url).netloc == domain and full_url not in visited_urls:
                urls_to_visit.append(full_url)

    #Loops through all links in a FIFO queue format
    while urls_to_visit:
        url = urls_to_visit.pop(0)
        web_crawl(url)


if __name__ == '__main__':
    url = urls_to_visit.pop(0)
    web_crawl(url)

