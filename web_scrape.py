import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

#Global Variables
start_url = "https://www.cornellcollege.edu"
domain = urlparse(start_url).netloc
visited_urls = set()
urls_to_visit = [start_url]


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

    #finds all
    tags = soup.find_all(["h2", "p"])

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
