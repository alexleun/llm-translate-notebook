**README.md for Trans2Chinese.py**

**Introduction**

Trans2Chinese.py is a Python program that allows you to translate text to Chinese using a local LLM server with API support. The local LLM should be run on the same computer, with the default address being:

"http://127.0.0.1:5000/v1/chat/completions"

**Features**

* Translates text to Chinese using a local LLM server
* Supports custom chunk size and language selection
* Option to keep the original text in the output file for learning purposes
* **New:** Supports PyTorch installation for GPU acceleration
* **New:** Supports convert to Tranditional Chinese
* **New:** Supports Option to keep original text in output file.


**Requirements**

* Python 3.6 or later
* langchain
* opencc
* tqdm
* argparse
* PyTorch (for GPU acceleration)

**Installation**

1. Install the required packages:

```
pip install langchain opencc tqdm argparse
```

2. Install PyTorch for GPU acceleration:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Clone this repository:

```
git clone https://github.com/alexleun/llm-translate-notebook.git
```

**Usage**

1. Open the `api_trans.py` script in a text editor.

2. Set the following parameters:

* `in_file`: Path to the input text file
* `out_file`: Path to the output text file
* `Translate_counter`: Starting translation counter (useful for resuming a stopped translation)
* `custom_chunk_size`: Size of each chunk of text to be translated (default: 250)
* `custom_language`: Language to translate to (default: "Chinese")
* `Keep_Orignial`: Whether to keep the original text in the output file (default: False)

3. Run the script:

```
python Trans2Chinese.py
```

**Example**

To translate the text in `input.txt` to Chinese and save the output in `output.txt`, run the following command:

```
python Trans2Chinese.py [-h] [--Translate_counter TRANSLATE_COUNTER] [--custom_chunk_size CUSTOM_CHUNK_SIZE]
                        [--custom_language CUSTOM_LANGUAGE] [--Keep_Orignial KEEP_ORIGNIAL]
                        in_file out_file
```

**Note:**

* The translation quality may vary depending on the complexity of the text and the chosen LLM model.
* If you encounter any errors, please check your LM Studio API key and ensure that you have sufficient GPU resources available for the selected model.


# Additionally install after spaCy: 

python -m spacy download en_core_web_sm


@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
