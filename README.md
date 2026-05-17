# llm-guide-translate

Translate large text files (novels, documents, etc.) using a local LLM via API (LM Studio / Ollama). Maintains a **translation guide** — an LLM-generated glossary — to keep terminology and names consistent across thousands of chunks.

## Features

- **Chunk-based translation** — splits large files into manageable chunks, translates each via local LLM API
- **Translation guide** — LLM analyzes a sample of the text to build a glossary (document type, writing style, term mappings) used in every translation prompt
- **Resume support** — saves progress to `-translation-status.json`; interrupted runs can resume with `-r`
- **Consistency** — the guide stores confirmed term/name translations (`confirmed_translations`) and includes them in each chunk's prompt
- **Flexible** — works with any language pair (default: → 繁體中文), any text domain
- **Echo tolerance** — when the model echoes the prompt instead of translating, the output is still written to the file for manual cleanup

## Requirements

- Python 3.8+
- A running LLM API endpoint (LM Studio, Ollama, or OpenAI-compatible)
- Dependencies in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

Edit the script variables at the top to match your API endpoint and model:

```python
url = "http://127.0.0.1:1234/v1/chat/completions"
model = "translategemma-4b-it"
```

## Usage

```bash
python main.py -i input.txt -l "繁體中文"
```

### Arguments

| Argument | Description |
|----------|-------------|
| `-i`, `--input_file` | Input text file to translate |
| `-c`, `--chunk_size` | Characters per chunk (default: 300) |
| `-l`, `--language` | Target language (default: 繁體中文) |
| `-r`, `--resume` | Resume from saved progress |

### Output files

For an input file `novel.txt`, the script generates:

| File | Description |
|------|-------------|
| `novel-big5.txt` | Translated output (UTF-8) |
| `novel-translation-guide.json` | LLM-generated glossary and confirmed translations |
| `novel-translation-status.json` | Progress tracker for resume |

## How it works

1. **Sample & analyze** — takes a sample of chunks, asks the LLM to identify the document type, writing style, and terminology suggestions
2. **Build guide** — stores the analysis as a `-translation-guide.json` file with confirmed term mappings
3. **Translate** — for each chunk, builds a prompt that includes the guide's name context and style guidelines
4. **Update guide** — extracts new terms from each chunk and appends them to the confirmed translations
5. **Resume** — on interruption, `-r` reloads the status file and continues from the last processed chunk

## Example

```bash
python main.py -i novel.txt -l "繁體中文" -c 300
python main.py -i novel.txt -l "繁體中文" -r   # resume after interruption
```

## License

MIT
