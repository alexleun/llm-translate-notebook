from langchain.prompts import PromptTemplate
import io
from langchain.text_splitter import CharacterTextSplitter
import requests
from tqdm import tqdm
import argparse
import opencc
import json
import os
import re
import hashlib
import time
from typing import Dict, List, Any
from datetime import datetime

# --- 全域變數 (可修改) ---
url = "http://127.0.0.1:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_HERE",
}
# model = "qwen/qwen3.5-9b"
model = "translategemma-4b-it"
converter = opencc.OpenCC('s2t')

# --- 設定值 ---
SUGGESTION_FILE_SUFFIX = "-translation-guide.json"
STATUS_FILE_SUFFIX = "-translation-status.json"
SAMPLE_SIZE = 10
REQUEST_TIMEOUT = 120  # 2 minutes per request
MAX_RETRIES = 5

# --- 分割文字成區塊 ---
def split_text(text, chunk_size=20):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

# --- 提取專有名詞 ---
def extract_terms(text: str) -> List[str]:
    terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    common = {'The', 'And', 'For', 'But', 'Not', 'You', 'Your', 'This', 'That', 'With', 'Chapter', 'Section'}
    return [t for t in terms if t not in common and len(t) > 2 and not t.isupper()]

# ============================================================
# 翻譯狀態管理
# ============================================================
def load_translation_status(input_file: str, chunk_size: int, resume: bool = False) -> Dict[str, Any]:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    status_file = f"{base_name}{STATUS_FILE_SUFFIX}"
    if resume and os.path.exists(status_file):
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        print(f"已加載翻譯狀態: {status_file}")
        print(f"  已處理: {status['processed_chunks']}/{status['total_chunks']}")
        return status
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = split_text(text, chunk_size)
    status = {
        "input_file": input_file, "chunk_size": chunk_size,
        "total_chunks": len(chunks), "processed_chunks": 0,
        "processed_chunk_indices": [], "version": "1.0",
        "created_at": datetime.now().isoformat(), "last_updated": datetime.now().isoformat()
    }
    _write_json(status_file, status)
    print(f"已建立翻譯狀態: {status_file}")
    return status

def save_translation_status(input_file: str, status: Dict[str, Any]):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    status["last_updated"] = datetime.now().isoformat()
    _write_json(f"{base_name}{STATUS_FILE_SUFFIX}", status)

# ============================================================
# 翻譯指南管理
# ============================================================
def load_translation_guide(input_file: str, chunks: List[str], status: Dict[str, Any], language: str) -> Dict[str, Any]:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    guide_file = f"{base_name}{SUGGESTION_FILE_SUFFIX}"
    if os.path.exists(guide_file) and status["processed_chunks"] > 0:
        guide = _read_json(guide_file)
        print(f"已載入翻譯指南: {guide_file}")
        print(f"  類型: {guide.get('llm_analysis', {}).get('document_type', '未知')}")
        print(f"  確認翻譯: {len(guide.get('confirmed_translations', {}))} 個")
        return guide
    print("正在用 LLM 分析文檔樣本，建立翻譯指南…")
    guide = create_translation_guide_with_llm(chunks, language)
    _write_json(guide_file, guide)
    print(f"已建立翻譯指南: {guide_file}")
    return guide

def save_translation_guide(input_file: str, guide: Dict[str, Any]):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    guide["updated_at"] = datetime.now().isoformat()
    _write_json(f"{base_name}{SUGGESTION_FILE_SUFFIX}", guide)

def parse_llm_json(text: str) -> dict:
    """嘗試從 LLM 回應中提取並修復 JSON。"""
    # 0) 先嘗試提取 markdown code block 內的 JSON
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    raw = m.group(1).strip() if m else text
    # 1) 找 {} 區塊（若 code block 提取後仍無效）
    if "{" not in raw:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("回應中無 {} 區塊")
        raw = m.group()
    else:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group()
    # 2) 嘗試解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # 3) 常見修復
    # 3a) 移除 // 註解行
    lines = raw.split("\n")
    cleaned = [l for l in lines if not l.strip().startswith("//")]
    raw = "\n".join(cleaned)
    # 3b) 移除尾逗號（陣列/物件最後項後的逗號）
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # 3c) 單引號轉雙引號（但避開已跳脫的）
    raw = re.sub(r"(?<!\\)'", '"', raw)
    # 3d) 補齊 key 的引號（key 沒引號時）
    raw = re.sub(r'(?<!")(\b[a-zA-Z_]\w*\b)(?=\s*:)', r'"\1"', raw)
    # 4) 再用 json.loads 嘗試
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # 失敗時顯示完整除錯
        print(f"  [除錯] LLM 完整回應 (前 2000 字元):")
        print(f"  {text[:2000]}")
        print(f"  [除錯] JSONDecodeError: {e}")
        raise e

def create_translation_guide_with_llm(chunks: List[str], language: str) -> Dict[str, Any]:
    if len(chunks) <= SAMPLE_SIZE:
        sample_indices = list(range(len(chunks)))
    else:
        step = len(chunks) // SAMPLE_SIZE
        sample_indices = sorted(set([0] + list(range(0, len(chunks), step))[:SAMPLE_SIZE-1]))
    samples = [chunks[i] for i in sample_indices]
    sample_text = "\n\n---\n\n".join(samples)
    all_terms = [t for s in samples for t in extract_terms(s)]
    unique_terms = list(dict.fromkeys(all_terms))[:30]
    
    example_json = """{"document_type": "輕小說", "writing_style": "輕鬆幽默", "main_themes": ["異世界", "冒險"], "tone_recommendation": "保持原文輕鬆語氣", "terminology_suggestions": [{"original": "ミリモス", "suggested_translation": "米洛莫斯", "reason": "音譯"}], "style_guidelines": ["保持原文格式"], "consistency_rules": ["專有名詞前後統一"]}"""

    prompt = f"""你是一位資深翻譯專家。請按照以下 JSON 格式回覆（terminology_suggestions 陣列中的每一項都必須是物件，不能是字串）：

範例格式：
{example_json}

字段說明：
- document_type: 文檔類型
- writing_style: 寫作風格
- main_themes: 主題列表
- tone_recommendation: 語氣建議
- terminology_suggestions: 術語翻譯建議陣列，每項必須是 {{"original": "...", "suggested_translation": "...", "reason": "..."}}
- style_guidelines: 風格指南列表
- consistency_rules: 一致性規則列表

樣本：
{sample_text}

潛在專有名詞：{', '.join(unique_terms) if unique_terms else '無'}

只輸出 JSON，不要輸出其他文字。"""
    default = {
        "document_type": "一般", "writing_style": "中性",
        "main_themes": ["一般"], "tone_recommendation": "保持原文風格",
        "terminology_suggestions": [],
        "style_guidelines": ["保持原文格式", "翻譯自然流暢"],
        "consistency_rules": ["專業術語前後一致"]
    }

    # 重試 3 次
    for attempt in range(3):
        try:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3 + attempt * 0.1, "max_tokens": 2000,
            }
            resp = requests.post(url, headers=headers, json=data, verify=False, timeout=90)
            resp.raise_for_status()
            body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise ValueError("choices 為空")
            text = body["choices"][0]["message"]["content"]
            analysis = parse_llm_json(text)

            # 驗證 terminology_suggestions 每項都是物件
            for item in analysis.get("terminology_suggestions", []):
                if not isinstance(item, dict) or "original" not in item:
                    raise ValueError("terminology_suggestions 中有無效項（不是物件或缺少 original）")
            break  # 成功
        except Exception as e:
            print(f"  嘗試 {attempt+1} 失敗: {e}")
            if attempt < 2:
                print(f"  重新嘗試…")
                time.sleep(2)
                continue
            print(f"  使用預設值")
            analysis = default

    confirmed = {}
    for item in analysis.get("terminology_suggestions", []):
        orig, sug = item.get("original", ""), item.get("suggested_translation", "")
        if orig and sug and orig != sug:
            confirmed[orig] = sug

    print(f"  LLM 分析完成 → {analysis.get('document_type', '一般')}")
    print(f"  建議翻譯: {len(confirmed)} 個")
    return {
        "llm_analysis": analysis, "confirmed_translations": confirmed,
        "term_frequency": {t: 0 for t in unique_terms},
        "quality_stats": {"total_chunks": 0, "valid_translations": 0},
        "sample_indices_used": sample_indices,
        "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat(),
        "version": "2.2"
    }

def update_translation_guide(original: str, translation: str, guide: Dict[str, Any]) -> Dict[str, Any]:
    terms = extract_terms(original)
    trans_terms = extract_terms(translation)
    for t in terms:
        guide["term_frequency"][t] = guide["term_frequency"].get(t, 0) + 1
    for orig, trans in zip(terms[:len(trans_terms)], trans_terms):
        if orig != trans and orig not in guide["confirmed_translations"]:
            guide["confirmed_translations"][orig] = trans
    guide["quality_stats"]["total_chunks"] += 1
    if translation != original:
        guide["quality_stats"]["valid_translations"] += 1
    return guide

# ============================================================
# 產生翻譯提示（恢復原始有效格式）
# ============================================================

def generate_translation_prompt(text: str, language: str, guide: Dict[str, Any]) -> str:
    a = guide.get("llm_analysis", {})
    confirmed = guide.get("confirmed_translations", {})

    name_context = ""
    if confirmed:
        items = list(confirmed.items())
        for orig, trans in items[:10]:
            name_context += f"  {orig} → {trans}\n"
        if len(items) > 10:
            name_context += f"  尚有 {len(items)-10} 個\n"

    template = """<start_of_turn>user
You are a professional to {language} translator. 
Your goal is to accurately convey the meaning and nuances of the original text while adhering to {language} grammar, vocabulary, and cultural sensitivities. 
Produce ONLY the {language} translation, without any additional explanations, introductory remarks, or commentary.

文檔類型：{doc_type}
寫作風格：{style}
{name_context}

Please translate the following text into {language}:
{question}<end_of_turn>
<start_of_turn>model
"""
    pt = PromptTemplate(template=template, input_variables=["question", "language", "doc_type", "style", "name_context"])
    return pt.format(
        question=text,
        language=language,
        doc_type=a.get("document_type", "一般"),
        style=a.get("writing_style", "中性"),
        name_context=name_context,
    )

# ============================================================
# 翻譯（Qwen 聊天格式）
# ============================================================
def get_translation(text: str, language: str, guide: Dict[str, Any]) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            prompt = generate_translation_prompt(text, language, guide)

            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "mode": "instruct",
                "instruction_template": "Alpaca",
                "temperature": 0.0,
            }

            resp = requests.post(url, headers=headers, json=data, verify=False, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise ValueError("choices 為空")

            raw = body["choices"][0]["message"]["content"].strip()

            # 顯示原始回應（前 200 字）以利除錯
            raw_preview = raw[:200].replace("\n", " ")
            print(f"  原始回應: {raw_preview}...")

            # 清除所有模板標記
            for tag in [
                "<start_of_turn>", "<end_of_turn>", "<|im_start|>", "<|im_end|>",
                "### Instruction:", "### Response:", "### Response",
                "### Human:", "### Assistant:", "### Output:", "</s>", "<s>",
            ]:
                raw = raw.replace(tag, "")

            # 清除殘留的 "model" 前綴（來自 <start_of_turn>model）
            raw = re.sub(r"^model\s*", "", raw)
            raw = re.sub(r"^\s*model\s*", "", raw)
            raw = re.sub(r"^<start_ofturn>model\s*", "", raw)

            # 清除常見前綴
            raw = re.sub(r"(?i)^(Translation|翻譯)[：:]?\s*", "", raw)
            raw = re.sub(r"(?i)^(Here is|Here's|The translation|Sure|Okay|Of course)[：:]?\s*", "", raw)
            raw = re.sub(r"(?i)^(好的|這是|這是翻譯|翻譯結果)[：:]?\s*", "", raw)
            raw = re.sub(r"^[【\[]?[Tt]ranslation[】\]]?\s*", "", raw)
            raw = re.sub(r"^[【\[]?翻譯[】\]]?\s*", "", raw)

            raw = raw.strip()

            # 若清理後為空，嘗試從原始輸出擷取中文字元
            if len(raw) < 3:
                chinese_only = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', raw_preview)
                if chinese_only:
                    fallback = "".join(chinese_only).strip()
                    print(f"  從原始回應擷取中文字: {fallback[:60]}...")
                    raw = fallback

            # 偵測模型是否只是回聲了提示文字（未實際翻譯）
            if len(raw) > 50:
                text_preview = text[:50].strip()
                if text_preview in raw:
                    print(f"  偵測到模型回聲提示文字（包含原文），仍寫入檔案（請手動檢查）")

            if len(raw) < 3:
                print(f"  嘗試 {attempt+1}: 輸出為空或過短，重試…")
                time.sleep(1)
                continue

            # 新增除錯：顯示回應摘要
            if attempt == 0:
                preview = raw[:80].replace("\n", " ")
                print(f"  回應預覽: {preview}...")

            return converter.convert(raw)

        except Exception as e:
            print(f"  嘗試 {attempt+1} 錯誤: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)

    print(f"  所有重試失敗，跳過此區塊")
    return None

# ============================================================
# 寫入結果
# ============================================================
def append_translation(input_file: str, text: str, chunk_index: int):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_file = f"{base_name}-big5.txt"
    mode = "a" if os.path.exists(out_file) else "w"
    with open(out_file, mode, encoding="UTF-8") as f:
        f.write(f"{text}\n")
        f.flush()

# ============================================================
# 主函數
# ============================================================
def translate_with_guide(text: str, input_file: str, language: str = "繁體中文",
                         chunk_size: int = 20, resume: bool = False):
    chunks = split_text(text, chunk_size)
    status = load_translation_status(input_file, chunk_size, resume)
    guide = load_translation_guide(input_file, chunks, status, language)

    print(f"開始翻譯，共 {len(chunks)} 區塊")
    start = status["processed_chunks"]
    if start > 0:
        print(f"從區塊 {start} 恢復")

    for i in tqdm(range(start, len(chunks)), desc="翻譯中", dynamic_ncols=True, initial=start, total=len(chunks)):
        chunk = chunks[i]

        # 跳過極小 / 純數字
        if chunk.isnumeric() or (len(chunk) == 29 and chunk[-3:].isnumeric()) or len(chunk) < 5:
            append_translation(input_file, chunk, i)
            status["processed_chunks"] = i + 1
            status["processed_chunk_indices"].append(i)
            save_translation_status(input_file, status)
            continue

        try:
            translation = get_translation(chunk, language, guide)
            if translation is None:
                print(f"  區塊 {i}: 翻譯失敗，跳過")
                status["processed_chunks"] = i + 1
                status["processed_chunk_indices"].append(i)
                save_translation_status(input_file, status)
                continue

            guide = update_translation_guide(chunk, translation, guide)
            append_translation(input_file, translation, i)

            status["processed_chunks"] = i + 1
            status["processed_chunk_indices"].append(i)
            save_translation_status(input_file, status)

            if (i + 1) % 5 == 0:
                save_translation_guide(input_file, guide)

            # 短延遲
            time.sleep(0.05)

        except Exception as e:
            print(f"\n區塊 {i} 錯誤: {e}")
            save_translation_status(input_file, status)
            save_translation_guide(input_file, guide)
            raise

    status["completed"] = True
    status["end_time"] = datetime.now().isoformat()
    save_translation_status(input_file, status)
    save_translation_guide(input_file, guide)

    s = guide["quality_stats"]
    print(f"\n完成！{s['total_chunks']} 區塊，{s['valid_translations']} 有效")

# ============================================================
# 工具函數
# ============================================================
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="翻譯工具 v2.2")
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-c", "--chunk_size", type=int, default=300)
    parser.add_argument("-l", "--language", default="繁體中文")
    parser.add_argument("-r", "--resume", action="store_true")

    args = parser.parse_args()

    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"錯誤: 找不到 {args.input_file}")
        sys.exit(1)

    print("=" * 50)
    print("  翻譯工具 v2.2")
    print("=" * 50)
    print(f"  檔案: {args.input_file}  ({len(text)} 字元)")
    print(f"  區塊: {args.chunk_size}")
    print(f"  語言: {args.language}")
    print(f"  恢復: {'是' if args.resume else '否'}")
    print("=" * 50)

    try:
        translate_with_guide(text, args.input_file, args.language, args.chunk_size, args.resume)
        base = os.path.splitext(os.path.basename(args.input_file))[0]
        print(f"\n輸出: {base}-big5.txt")
    except KeyboardInterrupt:
        print("\n\n中斷，可用 -r 恢復")
    except Exception as e:
        print(f"\n錯誤: {e}\n可用 -r 恢復")
