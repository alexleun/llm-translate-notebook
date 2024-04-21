from langchain.prompts import PromptTemplate
import io
from langchain.text_splitter import CharacterTextSplitter
import requests
from tqdm import tqdm
import argparse
import opencc
import json
import os
import logging
# to skip printing langchain message e.g. "Created a chunk of size 42, which is longer than the specified 0"
logging.getLogger().setLevel(logging.ERROR)

# --- Global Variables (can be modified) ---
url = "http://127.0.0.1:5000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
converter = opencc.OpenCC('s2t')

# --- Function to split text into chunks ---
def split_text(text, chunk_size=20):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# --- Function to generate translation prompt ---
def generate_translation_prompt(text, language="中文"):
    template = """
    翻译成{language}。 您应该仅回答翻译，如果不需要翻译，则仅回答原始字符串:
    {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["question", "language"])
    return prompt.format(question=text, language=language)

# --- Function to get initial translation from API ---
def get_initial_translation(prompt):
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "temperature": 0.01,
        "top_p": 0.95,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    translation = response.json()['choices'][0]['message']['content']
    return converter.convert(translation)

# --- Function to check if translation is good --- 
def is_good_translation(original, translation):
    check_prompt = f"Is the following translation of '{original}' accurate and fluent?\n\n{translation}"
    data = {
        "messages": [{"role": "user", "content": check_prompt}],
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    response_text = response.json()['choices'][0]['message']['content']
    return ("翻译非常准确" in response_text.lower() or 
            "翻译是准确的" in response_text.lower() or 
            "是的" in response_text.lower() or 
            "翻译准确" in response_text.lower() or  
            "沒有发现不准确" in response_text.lower() or 
            "yes" in response_text.lower() or  
            "accurate" in response_text.lower() or 
            "fluent" in response_text.lower())

# --- Function to get reason for dissatisfaction ---
def get_dissatisfaction_reason(original, translation):
    feedback_prompt = f"请分析以下“{original}”的翻译，找出其中不准确、不流畅或不相关的地方。\n\n{translation}\n您应该回答翻译不够好的原因，例如， 翻译不准确，因为它没有传达与原文相同的含义，或者翻译不流畅，因为它包含语法错误或不恰当的措辞。\n\n如果翻译准确，请回答准确。"
    data = {
        "messages": [{"role": "user", "content": feedback_prompt}],
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    return response.json()['choices'][0]['message']['content']

# --- Function to get improved translation ---
def get_improved_translation(prompt, previous_translation):
    feedback = get_dissatisfaction_reason(prompt, previous_translation)
    new_prompt = "请再翻译一次，因为 " + feedback
    data = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": previous_translation},
            {"role": "user", "content": new_prompt}
        ],
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    return converter.convert(response.json()['choices'][0]['message']['content'])

# --- Main Translation Function --- 
def translate_text(text, language="中文", keep_original=False, chunk_size=20):
    chunks = split_text(text, chunk_size)
    original_text_list = []
    translated_text_list = []
    for chunk in tqdm(chunks, desc="Translating", dynamic_ncols=True):
        prompt = generate_translation_prompt(chunk, language)
        translation = get_initial_translation(prompt)
        while not is_good_translation(chunk, translation):
            translation = get_improved_translation(prompt, translation)
        original_text_list.append(chunk)
        translated_text_list.append(translation)
    if keep_original:
        return list(zip(original_text_list, translated_text_list))
    else:
        return translated_text_list

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (parser arguments remain the same)
    parser.add_argument("-i", "--input_file", type=str, default="default.txt", help="Path to the input file")
    parser.add_argument("-c", "--chunk_size", type=int, default=20, help="Chunk size for text splitting")
    parser.add_argument("-l", "--language", type=str, default="中文", help="Language to translate to")
    parser.add_argument("-k", "--keep_original", action="store_true", help="Keep the original text in the output file")
    parser.add_argument("-t", "--translate_counter", type=int, default=0, help="Translate counter for restart from stop point")    

    args = parser.parse_args()
# ... (previous code: functions and argument parsing)

# Get the base file name without the extension
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    out_file = f"{base_name}-big5.txt"
    json_file = f"{base_name}-big5.json"
    translate_counter = 0
    translate_counter = args.translate_counter
    # print(f'\n\ntranslate_counter={translate_counter}\n\n')

    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = split_text(text, args.chunk_size)
    original_text_list = []
    translated_text_list = []

    with open(out_file, 'a', encoding='UTF-8') as outfile, open(json_file, 'a', encoding='UTF-8') as json_file:
        i = -1
        for chunk in tqdm(chunks, desc="Translating", dynamic_ncols=True):
            i += 1
            if i >= translate_counter:
            
                # print(f"\nchunk:'{chunk}'  is {not (chunk.isnumeric() or (len(chunk)==29 and chunk[-3:].isnumeric()) or len(chunk)<10)}\n")
                # We will not translate if chunked string is too small; is numeric; is timestamp
                if not (chunk.isnumeric() or (len(chunk)==29 and chunk[-3:].isnumeric()) or len(chunk)<5):
                    prompt = generate_translation_prompt(chunk, args.language)
                    translation = get_initial_translation(prompt)
                    while not is_good_translation(chunk, translation):
                        translation = get_improved_translation(prompt, translation)

                    # Save to text file
                    if args.keep_original:
                        outfile.write(f"\n{chunk}\n{translation}")
                    else:
                        outfile.write(f"\n{translation}")
                    #outfile.flush()  # Ensure data is written to disk

                    # Save to JSON file
                    original_text_list.append(chunk)
                    translated_text_list.append(translation)
                    json_data = {
                        "original_text": original_text_list,
                        "translated_text": translated_text_list
                    }
                    json.dump(json_data, json_file, ensure_ascii=False, indent=4)
                    json_file.write('\n')
                    #json_file.flush()
                    original_text_list=[]
                    translated_text_list=[]
                else:
                    outfile.write(f"\n{chunk}")