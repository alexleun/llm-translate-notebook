# all_json.py

import os
import argparse
from tqdm import tqdm
import json

# Import functions from Trans2Chinese-v5.py (assuming both files are in the same directory)
from Trans2Chinese import translate_text, split_text

# Global variables for API interaction
url = "http://127.0.0.1:5000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
#converter = opencc.OpenCC('s2t')  # Assuming you have opencc installed

def api_trans(in_file="temp.txt", out_file="temp-big5.txt", json_out_file="n4764du-qwen15.json", 
              Translate_counter=0, custom_chunk_size=20, language="Chinese", Keep_Orignial=False):

    with open(in_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\n\nTranslate:{in_file}\n\n")
    # Get translations (assuming translate_text returns a list of translations)
    translations = translate_text(text, language, Keep_Orignial, custom_chunk_size)

    # ... (rest of the code for printing information)

    
    # Save translations to files
    original_text_list = []
    translated_text_list = []
    with open(out_file, 'a', encoding='UTF-8') as outfile, open(json_out_file, 'a', encoding='UTF-8') as json_file:
        for i, translation in enumerate(translations):
            # Recreate original text chunks based on index and chunk size
            original_text = text[i*custom_chunk_size : (i+1)*custom_chunk_size]

            # Save to text file
            if Keep_Orignial:
                outfile.write(f"\n{original_text}\n{translation}")
            else:
                outfile.write(f"\n{translation}")
            outfile.flush()

            # Save to JSON file
            original_text_list.append(original_text)
            translated_text_list.append(translation)
            json_data = {
                "original_text": original_text_list,
                "translated_text": translated_text_list
            }
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
            json_file.write('\n')
            json_file.flush()
            original_text_list=[]
            translated_text_list=[]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Translate_counter", type=int, default=0, help="Translate counter value")
    parser.add_argument("--custom_chunk_size", type=int, default=20, help="custom chunk size")
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

        # Translate the txt file using imported function
        api_trans(txt_file, output_file, json_output_file, args.Translate_counter, args.custom_chunk_size,
                  args.custom_language, args.Keep_Orignial)

        # Update the progress bar for the number of files being processed
        p1.update(1)