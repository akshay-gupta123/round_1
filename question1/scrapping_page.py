import json
import requests
from bs4 import BeautifulSoup as bs
import numpy as np


url = input("Enter the url")
#url = "https://summerofcode.withgoogle.com/archive/2019/projects/"
load = requests.get(url)
soup = bs(load.text, "html.parser")

for i in soup.findAll('span',href = False):
     last_class =  i.get('class')
for i in soup.findAll('li',flex = True):
    flex = i.get('flex')
    break

pages = soup.find('span',attrs={'class':last_class})
pages = pages.text
pages = pages.split(" ")
pages = int(pages[3])



name =[]
project = []
organization = []


for i in soup.findAll('li',attrs={"flex":flex}):
    next = i.find('div')
    text = next.text.split("\n")
    name.append(text[2])
    project.append(text[4])
    org = text[5][14:]
    organization.append(org)
    #print(text[2],text[4],text[5])



for j in range(2,pages+1):
    #url = "https://summerofcode.withgoogle.com/archive/2019/projects/?page="+str(i)
    url+="?page="+str(i)
    load = requests.get(url)
    soup = bs(load.text, "html.parser")

    for i in soup.findAll('li', attrs={"flex": "50"}):
        next = i.find('div')
        text = next.text.split("\n")
        name.append(str(text[2]))
        project.append(str(text[4]))
        org = text[5][14:]
        organization.append(str(org))

np.savetxt('scrapping.csv', [p for p in zip(name,organization,project)], delimiter=',', fmt='%s',encoding='UTF-8')


