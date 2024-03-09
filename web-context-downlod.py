# Get the web contain in Japan novel, work with jp site
import requests
from bs4 import BeautifulSoup

def find_between(s, start, end):
    return s.split(start)[1].split(end)[0]

# Get subtitle and url from source_url
novel_num = "n0174gy"
base_url = "https://ncode.syosetu.com/"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(base_url+novel_num+'/', headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
novel_title = find_between(str(soup.find(class_="novel_title")), ">", "<")
index_box = soup.find(class_="index_box")
urls = index_box.find_all(class_="subtitle")
sublist = []
for url in urls:
    ut = find_between(str(url),'/">','</a>')
    ul = base_url+find_between(str(url),'href="/','">')
    sublist.append([ut, ul])

txt=novel_title+'\n'
e = 0
for subl in sublist:
    print(subl[1])
    e += 1
    txt=txt+'Chapter '+str(e)+': '+subl[0]+'\n'
    response = requests.get(subl[1], headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    a = soup.find_all(class_="novel_view")
    for b in a:
        c=b.get_text(separator = '</p>').replace('</p>','')
    txt=txt+c+'\n'

# write to text file.
text_file = open(novel_num+".txt", "w", encoding='UTF-8')
if text_file.write(txt) != len(txt):
    print("Failure! String not written to text file.")
text_file.close()
