# scrape government data from rice fondren website
# Import libraries
import requests
from bs4 import BeautifulSoup

# URL from which pdfs to be downloaded
url = "https://digitalcollections.rice.edu/documents/detail/report-to-the-president-on-u.s.-preparations-for-2009-h1n1-influenza/266404"

# Requests URL and get response object
response = requests.get(url)

# Parse text obtained
soup = BeautifulSoup(response.text, 'html.parser')

# Find all hyperlinks present on webpage
links = soup.find_all('a')
print(links)

i = 0

# From all links check for pdf link and
# if present download file
for link in links:
    if ('.pdf' in link.get('href', [])):
        print(link)
        i += 1
        print("Downloading file: ", i)

        # Get response object for link
        response = requests.get(link.get('href'))

        # Write content in pdf file
        pdf = open("pdf"+str(i)+".pdf", 'wb')
        pdf.write(response.content)
        pdf.close()
        print("File ", i, " downloaded")

print("All PDF files downloaded")

