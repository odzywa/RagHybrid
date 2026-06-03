import json
import re

from app.rag.ingest import ingest
from app.rag.smart_filter import classify_qa_batch, quick_classify_qa
from app.rag.generate import DEFAULT_MODEL


SMART_BATCH_SIZE = 12


def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text


def is_valid(text: str) -> bool:
    if not text:
        return False

    if len(text) < 30:
        return False

    return True


def message_text(message) -> str:
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        parts = content.get("parts", [])
        return "\n".join(part for part in parts if isinstance(part, str))

    if isinstance(content, list):
        return "\n".join(part for part in content if isinstance(part, str))

    return ""


def mapped_messages(conversation):
    mapping = conversation.get("mapping", {})

    messages = []
    for node in mapping.values():
        message = node.get("message") or {}
        author = message.get("author") or {}
        role = author.get("role")

        if role not in ("user", "assistant"):
            continue

        text = message_text(message)
        if text:
            messages.append({
                "role": role,
                "content": text,
                "create_time": message.get("create_time") or 0,
            })

    return sorted(messages, key=lambda item: item["create_time"])


def extract_qa_pairs(conversation):
    if not isinstance(conversation, dict):
        return []

    pairs = []

    messages = conversation.get("messages") or mapped_messages(conversation)

    for i in range(len(messages) - 1):
        m1 = messages[i]
        m2 = messages[i + 1]

        if m1.get("role") == "user" and m2.get("role") == "assistant":
            q = message_text(m1)
            a = message_text(m2)

            pairs.append((q, a))

    return pairs


def ingest_qa_pair(q: str, a: str, tags):
    tags = ["chatgpt"] + [
        tag for tag in tags
        if tag and tag != "chatgpt"
    ]

    text = f"""# ChatGPT Q/A

#{' #'.join(tags)}

## Pytanie
{q}

## Odpowiedź
{a}
"""

    ingest(text, source="chatgpt")


def import_chatgpt_json(path: str, smart: bool = False, backend: str = "auto", model: str = DEFAULT_MODEL):
    print(f"Importing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = data if isinstance(data, list) else [data]

    total = 0
    pending_llm = []

    def flush_pending_llm():
        nonlocal total, pending_llm

        if not pending_llm:
            return

        decisions = classify_qa_batch(
            pending_llm,
            backend=backend,
            model=model
        )
        print(f"SMART BATCH DECISIONS: {len(decisions)}")

        for item, decision in zip(pending_llm, decisions):
            print("SMART DECISION:", decision)

            if not decision.get("keep"):
                continue

            tags = item.get("hint_tags", []) + decision.get("tags", [])
            ingest_qa_pair(item["question"], item["answer"], tags)
            total += 1

        pending_llm = []

    for conv in conversations:
        pairs = extract_qa_pairs(conv)

        for q, a in pairs:
            q = clean_text(q)
            a = clean_text(a)

            if not is_valid(q) or not is_valid(a):
                continue

            tags = ["chatgpt"]

            if smart:
                quick = quick_classify_qa(q, a)
                print("SMART QUICK:", quick)

                if quick["decision"] == "reject":
                    continue

                if quick["decision"] == "keep":
                    ingest_qa_pair(q, a, quick.get("tags", []))
                    total += 1
                    continue

                pending_llm.append({
                    "id": len(pending_llm) + 1,
                    "question": q,
                    "answer": a,
                    "hint_tags": quick.get("tags", []),
                })

                if len(pending_llm) >= SMART_BATCH_SIZE:
                    flush_pending_llm()

                continue

            ingest_qa_pair(q, a, tags)

            total += 1

    flush_pending_llm()

    print(f"Imported Q/A pairs: {total}")
    return total
