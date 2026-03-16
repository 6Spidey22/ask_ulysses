import requests
from bs4 import BeautifulSoup


def html_store(url):
    #this function takes a url and grabs the html and writes it to the CornellCollege.html file
    html = requests.get(url)

    if html.status_code == 200:
        #if html.status_code == 200 that means the request was successful
        with open('CornellCollege.html', 'w', encoding='utf-8') as f:
            f.write(html.text)
        print("Successfully stored HTML in CornellCollege.html")

    else:
        raise Exception("Failed to store HTML in CornellCollege.html")

def fetch_links(soup):
    #this function finds all links stored in a BeautifulSoup object and stores them in a list
    links = [link.get('href') for link in soup.find_all('a')]
    return links


if __name__ == '__main__':
    url = "https://www.cornellcollege.edu/"
    html_store(url)

    #creating the BeautifulSoup object that will store the html information from CornellCollege.html
    with open('CornellCollege.html', 'r', encoding='utf-8') as f:
        html_doc = f.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    print(f"Page Title: {soup.title.string}")
