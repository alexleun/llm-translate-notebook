# Auto progress with all txt file in same folder.

import os
import argparse
from langchain.prompts import PromptTemplate
import io
from langchain.text_splitter import CharacterTextSplitter
import requests
from tqdm import tqdm
import opencc
import json

url = "http://127.0.0.1:5000/v1/chat/completions"
# url = "http://127.0.0.1:5000/v1/completions"
headers = {
    "Content-Type": "application/json"
}
history = []
converter = opencc.OpenCC('s2t')

def api_trans(in_file = "n4764du.txt", out_file = "n4764du-qwen15.txt", json_out_file = "n4764du-qwen15.json", Translate_counter = 0, custom_chunk_size = 250, language = "Chinese", Keep_Orignial = False):

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
    original_text_list = []
    translated_text_list = []

    p2 = tqdm(total=len(texts), desc="Translating", dynamic_ncols=True, position=0, leave=True)
    x = 0
    with open(out_file, 'a', encoding='UTF-8') as fp:
        for i in texts:
            #p2.value += 1
            p2.update(1)
            x += 1
            if x >= Translate_counter:
                # print(f"{prompt.format(question=i, language=language)}")
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
                fp.flush()
                original_text_list.append(i)
                translated_text_list.append(rs)
        fp.close()
        
    # Write the corrected JSON file
    corrected_json_data = {
        "original_text": original_text_list,
        "translated_text": translated_text_list
    }
    with open(json_out_file, 'w', encoding='UTF-8') as json_fp:
        json.dump(corrected_json_data, json_fp, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Translate_counter", type=int, default=0, help="Translate counter value")
    parser.add_argument("--custom_chunk_size", type=int, default=250, help="custom chunk size")
    parser.add_argument("--custom_language", default="Chinese", help="custom chunk size")
    parser.add_argument("--Keep_Orignial", type=bool, default=False, help="Keep original text or not")
    args = parser.parse_args()
    
    # Get the current working directory
    cwd = os.getcwd()

    # Get a list of all txt files in the directory
    txt_files = [f for f in os.listdir(cwd) if f.endswith(".txt")]

    # Create a progress bar for the number of files being processed
    p1 = tqdm(total=len(txt_files), desc="Files Processed", dynamic_ncols=True, position=1, leave=True)

    # Loop through each txt file
    for txt_file in txt_files:
        # Get the file name without the extension
        file_name = os.path.splitext(txt_file)[0]

        # Create the output file name
        output_file = file_name + "-big5.txt"
        json_output_file = file_name + "-big5.json"

        # Translate the txt file
        api_trans(txt_file, output_file, json_output_file, args.Translate_counter, args.custom_chunk_size,
        args.custom_language, args.Keep_Orignial)

        # Update the progress bar for the number of files being processed
        p1.update(1)