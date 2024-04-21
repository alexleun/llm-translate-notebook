## README.md for Trans2Chinese

### Introduction

Trans2Chinese-v5 is a Python program that allows you to translate text to Chinese using a local LLM server with API support. This version builds upon the previous iterations by incorporating new features and improvements:

**New Features:**

* **Traditional Chinese Conversion:** Allows conversion of translated text to Traditional Chinese using the OpenCC library.
* **Original Text Retention:** Provides the option to keep the original text in the output file for learning purposes.
* **Chunk Size Customization:** Enables users to specify the desired chunk size for text splitting, allowing for more granular control over the translation process.
* **Improved Translation Accuracy:** Leverages the power of the Qwen LLM model to deliver more accurate and nuanced translations.
* **Translation Quality Review:** Implements a feedback loop with the LLM to ensure translation accuracy.
* **Timestamp Detection:** For SRT files, the program automatically detects and skips timestamps. This ensures that only the dialogue content is translated, preserving the original timing and formatting of the subtitles.

**Key Features:**

* Translates text to Chinese using a local LLM server
* Supports custom chunk size and language selection
* Option to keep the original text in the output file for learning purposes
* Supports convert to Traditional Chinese
* Supports Option to keep original text in output file.
* Translation Quality Review for enhanced accuracy

### Requirements

* Python 3.6 or later
* langchain
* opencc
* tqdm
* argparse

### Installation

1. Install the required packages:

```
pip install langchain opencc tqdm argparse
```

2. Clone this repository:

```
git clone https://github.com/alexleun/llm-translate-notebook.git
```

### Usage

1. Open the `Trans2Chinese.py` script for single file translation or `all_json.py` for batch translation in a text editor.

2. Set the following parameters:

* `custom_chunk_size`: Size of each chunk of text to be translated (default: 20)
* `custom_language`: Language to translate to (default: "Chinese")
* `Keep_Orignial`: Whether to keep the original text in the output file (default: False)

3. Run the script:

**For single file translation:**

```
python Trans2Chinese.py [-h] [-i INPUT_FILE] [-c CHUNK_SIZE] [-l LANGUAGE] [-k] [-t TRANSLATE_COUNTER]

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Path to the input file
  -c CHUNK_SIZE, --chunk_size CHUNK_SIZE
                        Chunk size for text splitting
  -l LANGUAGE, --language LANGUAGE
                        Language to translate to
  -k, --keep_original   Keep the original text in the output file
  -t TRANSLATE_COUNTER, --translate_counter TRANSLATE_COUNTER
                        Translate counter for restart from stop point
```

**For batch translation:**

```
python all_json.py [-h] [--Translate_counter TRANSLATE_COUNTER] [--custom_chunk_size CUSTOM_CHUNK_SIZE]
                   [--custom_language CUSTOM_LANGUAGE] [--Keep_Orignial KEEP_ORIGNIAL]

options:
  -h, --help            show this help message and exit
  --custom_chunk_size CUSTOM_CHUNK_SIZE
                        custom chunk size
  --custom_language CUSTOM_LANGUAGE
                        custom chunk size
  --Keep_Orignial KEEP_ORIGNIAL
                        Keep original text or not
```

**Example:**

To translate the text in `input.txt` to Chinese and save the output in a file with the same name but with the added suffix "-big5.txt", run the following command:

```
python Trans2Chinese.py --in_file input.txt --Translate_counter 0 --custom_chunk_size 20 --custom_language Chinese --Keep_Orignial False
```

To translate all text files in the directory to Chinese and save the translated files in the same directory with the added suffix "-big5.txt", run the following command:

```
python all_json.py --custom_chunk_size 20 --custom_language Chinese --Keep_Orignial False
```

**Note:**

* The translation quality may vary depending on the complexity of the text and the chosen LLM model.
* If you encounter any errors, please check your LM Studio API key and ensure that you have sufficient GPU resources available for the selected model.

### Translation Quality Review

The Trans2Chinese-v5 program incorporates a translation quality review process. After the initial translation by the LLM, the translated text is sent back to the LLM for review. If the LLM deems the translation to be inaccurate, the process is repeated with a different translation model or parameters. This feedback loop ensures that the final output is of high quality and meets your expectations.

### Important Note

This project requires a local LLM server with API support, such as LM Studio or Oobabooga. Please ensure that you have a local LLM server set up and configured correctly before using this script.

**Recommendation:**

Based on my experience, using the "qwen 1.5 13B" LLM model from LM Studio provides the best results for translating text to Chinese. However, you may experiment with different LLM models and choose the one that delivers the most satisfactory results for your specific needs.

### Additional Information

* For GPU acceleration, ensure you have a compatible GPU and the necessary drivers installed.
* The `all_json.py` script automatically handles the translation process for multiple text files within a directory.
* The JSON output file provides a structured format for storing the original and translated text pairs.
* Trans2Chinese.py offers a convenient way to batch translate multiple text files.

### Contributing

We welcome contributions to this project. Please refer to the CONTRIBUTING.md file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.