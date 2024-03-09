# Trans2Chinese.py

Trans2Chinese.py is a stable Python program added for use with a local LLM server with API support. 

The local GPT should be run on the same computer, with the default address being:

"http://127.0.0.1:5000/v1/chat/completions"

Usage:

Trans2Chinese.py [-h] [--Translate_counter TRANSLATE_COUNTER] [--custom_chunk_size CUSTOM_CHUNK_SIZE] [--custom_language CUSTOM_LANGUAGE] in_file out_file

Where:

- custom_chunk_size is how long to chunk the original text file into segments for your local LLM to translate. Assume your local LLM cannot respond to a full document in one API query.

- Translate_counter is a segment number indicating the last segment translated, which can be used to continue the translation. But please use the previous chunk size, else it may miss something.  

The default prompt to the local LLM is:

template = """
Translate to Chinese: 
{question}
"""

It can be changed to support another language if the local LLM supports it.
I have tried many local LLMs and have found that qwen 1.5 72B provides the most useful responses.

# llm-translate-notebook

Jupyter Notebook Python script for translating text, ePubs and PDF files from English to Chinese.

Translation uses both GPU and CPU.

To use an NVIDIA GPU, install PyTorch:

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additionally install after spaCy: 

python -m spacy download en_core_web_sm


@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
