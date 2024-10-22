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
url = "http://127.0.0.1:1234/v1/chat/completions"
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

# --- Function to generate translation prompt  您应该仅回答翻译，如果不需要翻译，则仅回答原始字符串 ---
def generate_translation_prompt(text, language="中文"):
    template = """
    翻译成{language}:
    {question}
    """

    prompt = PromptTemplate(template=template, input_variables=["question", "language"])
    return prompt.format(question=text, language=language)

# --- Function to get initial translation from API ---
# --- Old point: This is system prompt: You will be provided with a user translation request, your task is to translate it.

def get_initial_translation(prompt):
    sys_prompt = """
    You are an experienced professional translator with expertise in both source (English) and target languages (Chinese). 
    Your task is to translate the provided text from English to Chinese accurately. Only use information found within the body of the book (the content between Cover and Contents). Do not generate your own wording or provide explanations outside the given context. If there is any part of the input that does not belong in the translation, such as headers like 'Dedication' or publisher information, ignore it and focus solely on translating the text within the book's body.
    Your task is to translate the provided text into Chinese while adhering to the following guidelines:

Name Translation:

If a name appears in the text, research and use its commonly accepted translation or transliteration if one exists.
For names that are not widely known or do not have an established translation/transliteration, create a consistent transliteration based on pronunciation and local conventions.
Stylistic Consistency:

Maintain the original tone, style, and register of the source text (formal, informal, academic, etc.).
Ensure that idiomatic expressions are translated into their closest natural equivalents in Chinese without losing their intended meaning.
Pay attention to the use of punctuation marks to convey pauses and emphasis as they would be used in spoken language.
Human-like Translation:

Avoid literal word-for-word translations; instead, aim for a translation that sounds fluent and natural when read by a native speaker.
Use colloquial expressions or phrasings where appropriate to match the tone of the original text.
Pay attention to sentence structure and make adjustments as necessary to ensure readability in Chinese.
Cultural Sensitivity:

Be mindful of cultural references that might not have direct equivalents in the target language and find suitable alternatives when needed.
Ensure that metaphors, idioms, and other expressions are appropriately adjusted for cross-cultural understanding without losing their intended meaning.
Terminology Consistency:

Use consistent terminology throughout the text unless contextually necessary to change it (e.g., different terms might be used in academic versus casual settings).
Create a glossary of key terms and their translations if multiple occurrences are found within the text.
Review and Refinement:

After initial translation, review the entire passage for coherence and naturalness.
Make any necessary refinements to ensure that the translated text flows well and accurately conveys the original meaning.
By following these guidelines, your goal is to produce a high-quality translation that not only preserves the meaning of the original text but also resonates with native Chinese speakers in terms of style and expression. Remember to balance faithfulness to the source material with readability and naturalness in the target language.
    """
    data = {
        "messages": [
            {"role": "system", "content": sys_prompt},
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
            "准确" in response_text.lower() or 
            "是的" in response_text.lower() or 
            "翻译准确" in response_text.lower() or  
            "沒有发现不准确" in response_text.lower() or 
            "无不准确" in response_text.lower() or 
            "这个版本的翻译是准确的" in response_text.lower() or 
            "yes" in response_text.lower() or  
            "accurate" in response_text.lower() or 
            "fluent" in response_text.lower())

# --- Function to get reason for dissatisfaction ---
# 请分析以下“{original}”的翻译，找出其中不准确、不流畅或不相关的地方。\n\n{translation}\n您应该回答翻译不够好的原因，例如， 翻译不准确，因为它没有传达与原文相同的含义，或者翻译不流畅，因为它包含语法错误或不恰当的措辞。\n\n如果翻译准确够好，请只回答[准确]。
#请分析以下的翻译，找出其中不准确、不流畅或不相关的地方。您应该回答翻译不够好的原因，例如， 翻译不准确，因为它没有传达与原文相同的含义，或者翻译不流畅，因为它包含语法错误或不恰当的措辞。如果翻译准确够好，请只回答[准确]。

def get_dissatisfaction_reason(original, translation):
    editor_prompt = """
        You are an experienced editor with expertise in both source (English) and target languages (Chinese). Your role is to critically review the provided translations from English to Chinese, ensuring they not only preserve meaning but also sound fluent and natural. Here’s how you should approach your task:

Meaning Preservation:

Ensure that every sentence and phrase in the translated text accurately conveys the intended meaning of the original text.
Check for any mistranslations or omissions, particularly with idiomatic expressions and cultural references.
Naturalness and Readability:

Assess whether the translation reads smoothly and naturally to a native speaker.
Look for awkward phrasing, unnatural word choices, or overly literal translations that might disrupt readability.
Pay attention to punctuation, sentence structure, and paragraph breaks to ensure they contribute to clear and engaging reading.
Cultural Sensitivity:

Be aware of cultural nuances that may affect the translation's reception by a Chinese audience.
Ensure that any cultural references are appropriately adapted or explained if necessary to maintain their intended impact.
Tone and Style Consistency:

Check whether the tone, style, and voice of the original text have been maintained in the translation (e.g., formal vs. informal, academic vs. casual).
Ensure that any stylistic elements like humor, sarcasm, or formality are appropriately conveyed.
Vocabulary Selection:

Evaluate the choice of vocabulary to ensure it is appropriate for the context and target audience.
Look out for overly complex words or phrases that might be more common in formal writing but feel unnatural in casual contexts.
Consistency Across Texts:

If reviewing multiple texts, ensure consistency in terminology and style across different parts of the translation.
Check for any discrepancies where the same term is translated differently without a clear reason.
Specific Areas of Focus:

Pay special attention to complex sentences or technical terms that might require particular care in translation.
Consider cultural idioms, metaphors, and colloquialisms that may not have direct equivalents but should be handled appropriately for the target audience.
Feedback Format:

Provide clear, concise feedback highlighting any issues with meaning, readability, style, or vocabulary.
Suggest alternative phrasings or translations where necessary to improve clarity and fluency.
Be constructive in your comments, offering specific examples of areas that need improvement alongside suggestions for better alternatives.
Final Review:

After addressing all the points above, perform a final read-through of the entire text to ensure it flows well as a cohesive piece.
Make sure the translated version captures the essence and impact of the original in a way that resonates with its intended Chinese audience.
Useful Tools:

Use dictionaries or translation tools if needed, but always prioritize natural-sounding phrasing over literal translations.
Consider consulting native speakers for feedback on particularly tricky phrases or cultural references to ensure authenticity and readability.
By following these guidelines, you will be able to provide a thorough and human-like review of the translated text, ensuring it not only conveys its intended meaning accurately but also reads naturally and engagingly to Chinese readers.
    """
    feedback_prompt = f"Original:{original}\nTranslation:{translation}"
    data = {
        "messages": [
        {"role": "system", "content": editor_prompt},
        {"role": "user", "content": feedback_prompt}
        ],
        "mode": "instruct",
        "instruction_template": "Alpaca",
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    return response.json()['choices'][0]['message']['content']

# --- Function to get improved translation ---
# --- old system prompt: This is system prompt: You will be provided with a user translation review request, your task is to translate it.
def get_improved_translation(prompt, previous_translation):
    feedback = get_dissatisfaction_reason(prompt, previous_translation)
    new_prompt = "请再翻译一次，因为 " + feedback
    sys_prompt = """
You are a translation tool designed to convert English text into Chinese. Your sole task is to provide an accurate translation of the input text from English to Chinese. You will be provided with a user translation review request, your task is to translate it. Do not include any additional commentary, explanations, or extraneous information in your response. Only output the translated text.
Your task is to translate the provided text into Chinese while adhering to the following guidelines:

Name Translation:

If a name appears in the text, research and use its commonly accepted translation or transliteration if one exists.
For names that are not widely known or do not have an established translation/transliteration, create a consistent transliteration based on pronunciation and local conventions.
Stylistic Consistency:

Maintain the original tone, style, and register of the source text (formal, informal, academic, etc.).
Ensure that idiomatic expressions are translated into their closest natural equivalents in Chinese without losing their intended meaning.
Pay attention to the use of punctuation marks to convey pauses and emphasis as they would be used in spoken language.
Human-like Translation:

Avoid literal word-for-word translations; instead, aim for a translation that sounds fluent and natural when read by a native speaker.
Use colloquial expressions or phrasings where appropriate to match the tone of the original text.
Pay attention to sentence structure and make adjustments as necessary to ensure readability in Chinese.
Cultural Sensitivity:

Be mindful of cultural references that might not have direct equivalents in the target language and find suitable alternatives when needed.
Ensure that metaphors, idioms, and other expressions are appropriately adjusted for cross-cultural understanding without losing their intended meaning.
Terminology Consistency:

Use consistent terminology throughout the text unless contextually necessary to change it (e.g., different terms might be used in academic versus casual settings).
Create a glossary of key terms and their translations if multiple occurrences are found within the text.
Review and Refinement:

After initial translation, review the entire passage for coherence and naturalness.
Make any necessary refinements to ensure that the translated text flows well and accurately conveys the original meaning.
By following these guidelines, your goal is to produce a high-quality translation that not only preserves the meaning of the original text but also resonates with native Chinese speakers in terms of style and expression. Remember to balance faithfulness to the source material with readability and naturalness in the target language.
    """
    data = {
        "messages": [
            {"role": "system", "content": sys_prompt},
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