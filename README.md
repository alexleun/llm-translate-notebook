# llm-translate-notebook
Jupyter Notebook python script for Translate text, epub and PDF file from English to Chinese.
Translation using GPU + CPU combine.

To use NVIDA GPU, should install tourch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

addidational install after spacy installed:
python -m spacy download en_core_web_sm
