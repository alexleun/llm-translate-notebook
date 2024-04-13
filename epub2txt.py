import os
from langchain.document_loaders import UnstructuredEPubLoader
from tqdm import tqdm

# Get all epub files in the current directory
epub_files = [f for f in os.listdir('.') if f.endswith('.epub')]

# Convert each epub file to a txt file
for epub_file in tqdm(epub_files, position=0, leave=True):
    filename = epub_file.replace('.epub', '')
    outfile = filename + '.txt'
    loader = UnstructuredEPubLoader(epub_file, mode="elements")
    data = loader.load()

    with open(outfile, 'a', encoding='UTF-8') as fp:
        for i in data:
            fp.write(i.dict()['page_content'])
            fp.write("[]")
        fp.close()