"""HotpotQA 原生 yes/no：保留 distractor 输入，不使用 gold supporting facts。"""
import random

from storage import DATA_ROOT, download, fingerprint

REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
SOURCE_URL = (f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{REVISION}/"
              "distractor/validation-00000-of-00001.parquet")
SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
HOME = "https://hotpotqa.github.io/"


def hotpot_context(context):
    """只接收段落字段：标题 + 原顺序句子；不接收问题、答案或 supporting_facts。"""
    titles, paragraphs = context["title"], context["sentences"]
    if not titles or len(titles) != len(paragraphs):
        raise ValueError("Empty or mismatched HotpotQA paragraph fields.")
    blocks = []
    for title, sentences in zip(titles, paragraphs):
        if not sentences or not all(isinstance(s, str) for s in sentences):
            raise ValueError("Malformed HotpotQA paragraph.")
        blocks.append(title + "\n" + " ".join(s.strip() for s in sentences))
    return "\n\n".join(blocks)


def convert_rows(rows):
    """先筛原生标签，再统一格式；不根据长度、类别比例或模型结果挑选。"""
    examples = []
    for index, row in enumerate(rows):
        answer = row["answer"].strip().lower()
        if answer not in {"yes", "no"}:
            continue
        context = hotpot_context(row["context"])
        examples.append({
            "id": f"hotpotqa-dev-{row['id']}", "source_row_zero_based": index,
            "context": context, "context_group": fingerprint(context),
            "question": row["question"], "label": int(answer == "yes"),
            "question_type": row["type"], "paragraph_count": len(row["context"]["title"]),
        })
    if len({e["id"] for e in examples}) != len(examples):
        raise ValueError("Duplicate HotpotQA IDs.")
    return examples


def load_pool(offline=False):
    # 延迟导入：BoolQ 和 QASPER 不需要这个可选依赖。
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("HotpotQA requires: python -m pip install -r requirements-data.txt") from error
    path = download(SOURCE_URL, DATA_ROOT / "hotpotqa/validation.parquet", SOURCE_SHA256, offline)
    rows = parquet.read_table(path).to_pylist()
    examples = convert_rows(rows)
    provenance = {
        "name": "HotpotQA yes/no (distractor)", "split": "dev", "dataset_home": HOME,
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256, "revision": REVISION,
        "source_format": "Hugging Face Parquet mirror; dev is named validation",
        "raw_rows": len(rows), "available_rows": len(examples), "license": "CC BY-SA 4.0",
        "selection_rule": "answer.strip().lower() in {yes, no}; no balancing or outcome filtering",
        "context_definition": "All supplied titles and paragraphs (usually 10), original order; sentences joined by spaces. No supporting_facts.",
        "scope": "The supplied retrieved context is question-conditioned; only the compressor is query-blind.",
    }
    return examples, provenance



"""QASPER v0.3：全文输入、严格一致的二元标签、可选文本证据筛选。"""
import json
import random
import tarfile
from collections import Counter

from storage import DATA_ROOT, download, fingerprint

SOURCES = {
    "train_dev": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
                  "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"),
    "test": ("https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
             "72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72"),
}


def load_papers(split, offline=False):
    if split not in {"train", "dev", "test"}:
        raise ValueError("QASPER split must be train, dev or test.")
    url, sha = SOURCES["test" if split == "test" else "train_dev"]
    path = download(url, DATA_ROOT / "qasper" / url.rsplit("/", 1)[1], sha, offline)
    member = f"qasper-{split}-v0.3.json"
    with tarfile.open(path, "r:gz") as archive:
        papers = json.load(archive.extractfile(member))
    return papers, {"source_url": url, "source_sha256": sha, "archive_member": member,
                    "license": "CC BY 4.0", "version": "0.3"}


def qasper_context(paper):
    """全文候选输入：标题、摘要、所有章节和段落；不附加问答或 gold evidence。"""
    parts = [paper["title"], paper["abstract"]]
    for section in paper["full_text"]:
        parts.append(section["section_name"] or "")
        parts.extend(section["paragraphs"])
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def answer_type(answer):
    """遵循官方 evaluator 的优先级；False 是 No，不是缺失值。"""
    if answer["unanswerable"]:
        return "unanswerable"
    if answer["extractive_spans"]:
        return "extractive"
    if answer["free_form_answer"]:
        return "abstractive"
    if type(answer["yes_no"]) is bool:
        return "boolean"
    return "invalid"


def classify_question(question):
    """严格一致子集：所有标注均是 boolean 且同标签，不投票、不挑第一份答案。"""
    answers = [annotation["answer"] for annotation in question["answers"]]
    types = [answer_type(a) for a in answers]
    if not answers:
        return "no_annotations", None
    if "boolean" not in types:
        return "no_boolean", None
    if any(t != "boolean" for t in types):
        return "mixed_answer_types", None
    labels = {a["yes_no"] for a in answers}
    if len(labels) != 1:
        return "conflicting_boolean_labels", None
    return "eligible", int(next(iter(labels)))


def text_evidence_status(question, paper):
    """额外保守检查，所有标注的证据需在全文中；不裁剪正文，会引入选择偏差。"""
    evidence_lists = [a["answer"]["evidence"] for a in question["answers"]]
    if any(e.startswith("FLOAT SELECTED") for items in evidence_lists for e in items):
        return "figure_or_table"
    if not evidence_lists or any(not items for items in evidence_lists):
        return "missing_evidence"
    text = " ".join(qasper_context(paper).split())
    if any(not e.strip() or " ".join(e.split()) not in text for items in evidence_lists for e in items):
        return "evidence_not_in_text"
    return "text_verified"


def convert_papers(papers, split, require_text_evidence=True):
    """证据只用于定义评测子集，不用于选择、标记或裁剪输入正文。"""
    examples, excluded = [], Counter()
    for paper_id, paper in papers.items():
        has_body = any(p.strip() for s in paper['full_text'] for p in s['paragraphs'])
        context = qasper_context(paper)
        for question in paper['qas']:
            status, label = classify_question(question)
            if status == 'eligible' and not has_body:
                status = 'missing_full_text'
            if status != 'eligible':
                excluded[status] += 1
                continue
            evidence = text_evidence_status(question, paper)
            if require_text_evidence and evidence != 'text_verified':
                excluded[evidence] += 1
                continue
            examples.append({
                'id': f"qasper-{split}-{question['question_id']}",
                'paper_id': paper_id, 'context': context,
                'context_group': fingerprint(context),
                'question': question['question'], 'label': label,
                'evidence_status': evidence,
            })
    if len({e['id'] for e in examples}) != len(examples):
        raise ValueError('Duplicate QASPER question IDs.')
    return examples, dict(excluded)



def load_examples(name, size, seed, offline):
    """name：数据集；size：题数；seed：抽样种子；offline：禁止下载。"""
    if name == 'hotpotqa':
        pool, source = load_pool(offline)
    elif name == 'qasper':
        papers, source = load_papers('train', offline)
        pool, exclusions = convert_papers(papers, 'train', require_text_evidence=True)
        source = {**source, 'split': 'train', 'excluded': exclusions,
                  'selection_rule': 'All boolean labels agree; all evidence occurs in full text.'}
    else:
        raise ValueError('Unsupported dataset: '+name)
    if not 1 <= size <= len(pool):
        raise ValueError(f'{name}: requested {size}, eligible pool has {len(pool)}.')
    # 按 ID 排序后抽样，复现 mini100 的选题顺序，而非依赖旧预测文件。
    examples = random.Random(seed).sample(sorted(pool, key=lambda e: e['id']), size)
    return examples, {**source, 'eligible_pool': len(pool), 'sample_seed': seed,
                      'sampling': 'uniform sample without replacement from sorted IDs'}

from tokenizers import Tokenizer
TOKENIZER_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
TOKENIZER_URL = f"https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/resolve/{TOKENIZER_REVISION}/tokenizer.json"
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

class ContextTokenizer:
    def __init__(self):
        self.path = download(
            TOKENIZER_URL, DATA_ROOT / "tokenizer" / "qwen3-30b-a3b-instruct-2507.json", TOKENIZER_SHA256
        )
        self.tokenizer = Tokenizer.from_file(str(self.path))

    def count(self, text):
        # 不包括 system prompt、问题或 chat template；API usage 另行记录。
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

