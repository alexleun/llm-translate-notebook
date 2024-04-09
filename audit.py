from langchain.prompts import PromptTemplate
import io
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
converter = opencc.OpenCC('t2s')

def audit_and_correct(in_file = "n4764du-big5.json", out_file = "n4764du-big5-checked.txt", json_out_file = "n4764du-big5-checked.json", language = "Chinese"):
    # Get the base file name without the extension
    base_name = os.path.splitext(os.path.basename(in_file))[0]

    original_text_list = []
    translated_text_list = []

    with open(in_file, 'r', encoding='UTF-8') as json_fp:
        for line in json_fp:
            try:
                data = json.loads(line)
                original_text_list.append(data['original_text'])
                translated_text_list.append(data['translated_text'])
            except json.JSONDecodeError:
                pass

    corrected_text_list = []
    corrected_translated_list = []

    p2 = tqdm(total=len(original_text_list), desc="Auditing", dynamic_ncols=True, position=0, leave=True)
    for original, translated in zip(original_text_list, translated_text_list):
        p2.update(1)
        prompt = f"""
        Please review the following translation and provide a corrected version if needed:

        Original text: {original}
        Translated text: {translated}

        Provide a corrected translation if needed, otherwise just say "No correction needed".
        """
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "mode": "instruct",
            "instruction_template": "Alpaca",
            "temperature": 0.7,
            "top_p": 0.9,
        }
        response = requests.post(url, headers=headers, json=data, verify=False)
        corrected_translation = response.json()['choices'][0]['message']['content']

        if corrected_translation.strip().lower() == "no correction needed":
            corrected_text_list.append(original)
            corrected_translated_list.append(translated)
        else:
            corrected_text_list.append(original)
            corrected_translated_list.append(corrected_translation)

    # Write the corrected text file
    with open(out_file, 'w', encoding='UTF-8') as fp:
        for original, corrected in zip(corrected_text_list, corrected_translated_list):
            if original != corrected:
                fp.write(f"\nOriginal: {original}\nCorrected: {corrected}\n")
            else:
                fp.write(f"\n{original}\n{corrected}\n")

    # Write the corrected JSON file
    corrected_json_data = {
        "original_text": corrected_text_list,
        "translated_text": corrected_translated_list
    }
    with open(json_out_file, 'w', encoding='UTF-8') as json_fp:
        json.dump(corrected_json_data, json_fp, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="input JSON file path")
    parser.add_argument("--out_file", help="output text file path")
    parser.add_argument("--json_out_file", help="output JSON file path")
    parser.add_argument("--custom_language", default="Chinese", help="custom language")
    args = parser.parse_args()
    audit_and_correct(args.in_file, args.out_file, args.json_out_file, args.custom_language)