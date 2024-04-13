from langchain.prompts import PromptTemplate
import io
from langchain.text_splitter import CharacterTextSplitter
import requests
from tqdm import tqdm
import argparse
import opencc
import json
import os

url = "http://127.0.0.1:5000/v1/chat/completions"
# url = "http://127.0.0.1:5000/v1/completions"
headers = {
    "Content-Type": "application/json"
}
history = []
converter = opencc.OpenCC('s2t')

# Translate_counter for user input if program has stop with restart from stop point
def api_trans(in_file = "n4764du.txt", Translate_counter = 0, custom_chunk_size = 250, language = "Chinese", Keep_Orignial = False):
    # Get the base file name without the extension
    base_name = os.path.splitext(os.path.basename(in_file))[0]

    out_file = f"{base_name}-big5.txt"
    json_file = f"{base_name}-big5.json"

    f = io.open(in_file, mode="r", encoding="utf-8")
    s = f.read()

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = custom_chunk_size,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.split_text(s)

    template = """
    Translate to {language}:
    {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    print(f"File Language Translate to {language}")
    print(f"Orignial file divided to {len(texts)} segment.")

    result=[]
    p2 = tqdm(total=len(texts), desc="Translating", dynamic_ncols=True, position=0, leave=True)
    x = 0
    original_text_list = []
    translated_text_list = []
    with open(out_file, 'a', encoding='UTF-8') as fp, open(json_file, 'a', encoding='UTF-8') as json_fp:
        for i in texts:
            p2.update(1)
            x += 1
            if x >= Translate_counter:
                data = {
                        "messages": [
                              {
                                "role": "user",
                                "content": prompt.format(question=i, language=language)
                              }
                                    ],
                        "mode": "instruct", #instruct
                        "instruction_template": "Alpaca",
                        "temperature": 0.7,
                        "top_p": 0.9,
                        }
                response = requests.post(url, headers=headers, json=data, verify=False)
                rs = response.json()['choices'][0]['message']['content']
                rs = converter.convert(rs)
                if Keep_Orignial:
                    fp.write('\n'+ i + '\n' + rs)
                else:
                    fp.write('\n'+ rs)
                original_text_list.append(i)
                translated_text_list.append(rs)

                # Write the JSON file
                json_data = {
                    "original_text": original_text_list,
                    "translated_text": translated_text_list
                }
                json.dump(json_data, json_fp, ensure_ascii=False, indent=4)
                json_fp.write('\n')
                json_fp.flush()
                original_text_list=[]
                translated_text_list=[]
                fp.flush()

        fp.close()
        json_fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="input file path")
    parser.add_argument("--Translate_counter", type=int, default=0, help="Translate counter value")
    parser.add_argument("--custom_chunk_size", type=int, default=250, help="custom chunk size")
    parser.add_argument("--custom_language", default="Chinese", help="custom chunk size")
    parser.add_argument("--Keep_Orignial", type=bool, default=False, help="Keep original text or not")
    args = parser.parse_args()
    api_trans(args.in_file, args.Translate_counter, args.custom_chunk_size,
    args.custom_language, args.Keep_Orignial)