import json
import hashlib
import re
from psycopg2 import pool

from app.config import settings
from app.rag.embed import embed_text
from app.rag.chunk import split_code_file, split_markdown, extract_tags, infer_command_tags, merge_tags

connection_pool = pool.SimpleConnectionPool(
    1,
    10,
    dsn=settings.DATABASE_URL
)


def get_conn():
    return connection_pool.getconn()


def source_count(source: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            "SELECT COUNT(*) FROM documents WHERE metadata->>'source' = %s",
            (source,)
        )
        return cur.fetchone()[0]

    finally:
        cur.close()
        connection_pool.putconn(conn)


def delete_source(source: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    try:
        cur.execute(
            "DELETE FROM documents WHERE metadata->>'source' = %s",
            (source,)
        )
        deleted = cur.rowcount
        conn.commit()
        return deleted

    finally:
        cur.close()
        connection_pool.putconn(conn)


def normalize(text: str) -> str:
    text = text.strip()
    text = text.replace("\r\n", "\n")

    # usuń wielokrotne spacje i taby
    text = re.sub(r"[ \t]+", " ", text)

    # usuń wielokrotne nowe linie
    text = re.sub(r"\n+", "\n", text)

    return text


def is_valid_chunk(text: str) -> bool:
    if not text:
        return False

    # minimalna długość
    if len(text) < 20:
        return False

    # za dużo znaków specjalnych = śmieci (PDF)
    special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    ratio = special / len(text)

    if ratio > 0.4:
        return False

    return True


def ingest(text, source="manual", metadata=None, chunk_mode="markdown"):
    print("INGEST FILE")

    metadata = metadata or {}
    chunks = split_code_file(text, filename=source) if chunk_mode == "code" else split_markdown(text)
    metadata_tags = metadata.get("tags", [])
    for key in ["collection", "language", "source_type"]:
        if metadata.get(key):
            metadata_tags.append(metadata[key])

    document_tags = merge_tags(
        metadata_tags,
        extract_tags(text[:3000]),
        infer_command_tags(text[:3000])
    )
    print(f"CHUNKS COUNT: {len(chunks)}")
    print("DOCUMENT TAGS:", document_tags)

    if chunks:
        print("FIRST CHUNK:", chunks[0][:100])

    conn = get_conn()
    cur = conn.cursor()
    stats = {
        "source": source,
        "chunks": len(chunks),
        "valid": 0,
        "inserted": 0,
        "duplicates": 0,
        "invalid": 0,
        "errors": 0
    }

    for chunk in chunks:
        try:
            normalized = normalize(chunk)
            valid = is_valid_chunk(normalized)

            print("CHUNK LEN:", len(normalized))
            print("VALID:", valid)
            print("PREVIEW:", normalized[:100])

            if not valid:
                stats["invalid"] += 1
                continue

            stats["valid"] += 1
            doc_hash = hashlib.md5(normalized.lower().encode()).hexdigest()

            tags = merge_tags(
                document_tags,
                extract_tags(normalized),
                infer_command_tags(normalized)
            )

            page_match = re.search(r"(?im)^##\s+Page\s+(\d+)", normalized)
            page = int(page_match.group(1)) if page_match else None
            chunk_metadata = {
                **metadata,
                "source": source,
                "tags": tags,
                "page": page
            }

            embedding = embed_text(normalized)

            cur.execute(
                """
                INSERT INTO documents (content, embedding, metadata, hash)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (hash) DO NOTHING
                """,
                (
                    normalized,
                    embedding,
                    json.dumps(chunk_metadata),
                    doc_hash
                )
            )

            if cur.rowcount == 1:
                stats["inserted"] += 1
            else:
                stats["duplicates"] += 1

        except Exception as e:
            print("CHUNK ERROR:", e)
            stats["errors"] += 1

    conn.commit()
    cur.close()
    connection_pool.putconn(conn)

    print("FILE PROCESSED:", stats)
    return stats
