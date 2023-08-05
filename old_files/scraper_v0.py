import requests
from bs4 import BeautifulSoup
import re
import urllib.parse

def get_soup(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    return soup

def get_links(soup, base_url):
    links = []
    for link in soup.find_all('a', href=True):
        url = link['href']
        # Resolve relative links
        url = urllib.parse.urljoin(base_url, url)
        # Only add links that start with the base URL
        if url.startswith(base_url) and url not in links:
            links.append(url)
    return links

def write_text(soup, file):
    # Ignore non-text content by only extracting text within <p> tags
    for paragraph in soup.find_all('p'):
        file.write(paragraph.get_text() + '\n')

def scrape_website(base_url, output_file):
    visited = set()
    to_visit = [base_url]
    
    with open(output_file, 'w') as f:
        while to_visit:
            url = to_visit.pop()
            if url in visited:
                continue
            visited.add(url)
            
            soup = get_soup(url)
            write_text(soup, f)
            
            links = get_links(soup, base_url)
            to_visit.extend(link for link in links if link not in visited)

base_url = "https://www.uscis.gov/working-in-the-united-states"
output_file = "scraped_text.txt"
scrape_website(base_url, output_file)