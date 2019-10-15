from bs4 import BeautifulSoup
import requests
import re
import os

url = "https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"
physionet_bash_dir = 'raw'
physionet_bash_path = physionet_bash_dir + '/physionet_bash.sh'

os.makedirs(physionet_bash_dir, exist_ok=True)

# Getting the webpage, creating a Response object.
response = requests.get(url)

# Extracting the source code of the page.
data = response.text

# Passing the source code to BeautifulSoup to create a BeautifulSoup object for it.
soup = BeautifulSoup(data, 'lxml')

# Extracting all the <a> tags into a list.
tags = soup.find_all('a')

# Extracting URLs from the attribute href in the <a> tags.
i, physionet_bash = 0, open(physionet_bash_path,'w')
for tag in tags:
    if re.findall('1E\w-\w*.edf',tag.get('href')) and i<80: # get top 40 patients night 1 data
        physionet_bash.write('wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/{}\n'.format(tag.get('href')))
        i = i+1

physionet_bash.close()