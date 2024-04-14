import requests
from bs4 import BeautifulSoup
import sys

def find_between(s, start, end):
    return s.split(start)[1].split(end)[0]

# Get subtitle and url from source_url
base_url = sys.argv[1]
novel_num = sys.argv[2]
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

page_num = 1
e = 0  # Initialize the variable 'e' outside the loop
txt = ""  # Initialize the variable 'txt' outside the loop
with open(novel_num + ".txt", "a", encoding='UTF-8') as fp:
    while True:
        response = requests.get(base_url + novel_num + '/?p=' + str(page_num), headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Check if there are any URLs on the current page
        index_box = soup.find(class_="index_box")
        try:
            urls = index_box.find_all(class_="subtitle")
            if not urls:
                break
        except:
            break

        # Process the URLs on the current page
        for url in urls:
            ut = find_between(str(url), '/">', '</a>')
            ul = base_url + find_between(str(url), 'href="/', '">')

            print(ul)

            e += 1
            txt = txt + 'Chapter ' + str(e) + ': ' + ut + '\n'
            response = requests.get(ul, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            a = soup.find_all(class_="novel_view")
            for b in a:
                c = b.get_text(separator='</p>').replace('</p>', '')
                txt = txt + c + '\n'
        page_num += 1
        fp.write(txt)
        txt = ""
fp.close()

#Write to text file.
# text_file = open(novel_num + ".txt", "a", encoding='UTF-8')
# if text_file.write(txt) != len(txt):
    # print("Failure! String not written to text file.")
# text_file.close()