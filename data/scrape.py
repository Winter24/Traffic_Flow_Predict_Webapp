import requests
from bs4 import BeautifulSoup

# URL you want to scrape (use an appropriate URL that allows scraping)
url = 'https://www.google.com/maps/d/u/0/viewer?entry=yt&mid=1nSTYcRSxiMOAUu1BsFtBAnshc3oIYaRz&ll=10.809926935393246%2C106.64760349999999&z=11'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all <div> tags in the parsed HTML
divs = soup.find_all('div')
with open('div_contents.txt', 'w', encoding='utf-8') as file:
    for div in divs:
        file.write(str(div))
        file.write("\n----------\n")  # Separate div contents for readability

print("Content has been written to div_contents.txt")