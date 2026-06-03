import os
from app.rag.ingest import ingest


def ingest_folder(path="/space"):
    print(f"Ingesting folder: {path}")

    for root, _, files in os.walk(path):
        for file in files:

            if not file.endswith(".md"):
                continue

            if file.startswith("index") or file.startswith("CONFIG"):
                continue

            filepath = os.path.join(root, file)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            print(f"Loading: {filepath}")

            ingest(content, source=file)