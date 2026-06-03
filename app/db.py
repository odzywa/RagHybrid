import time
import psycopg2
from app.config import settings


def get_db_connection():
    return psycopg2.connect(settings.DATABASE_URL)


def init_db():
    for i in range(10):
        try:
            conn = psycopg2.connect(settings.DATABASE_URL)
            cur = conn.cursor()

            # pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # tabela
            cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR(768),
                metadata JSONB,
                hash TEXT
            );
            """)

            # 🔥 migracje
            cur.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS hash TEXT;
            """)

            cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash
            ON documents(hash);
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS graph_index_status (
                document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                processed_at TIMESTAMP DEFAULT NOW(),
                relations_count INTEGER DEFAULT 0,
                new_relations_count INTEGER DEFAULT 0,
                existing_relations_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'done',
                error TEXT
            );
            """)

            cur.execute("""
            ALTER TABLE graph_index_status
            ADD COLUMN IF NOT EXISTS new_relations_count INTEGER DEFAULT 0;
            """)

            cur.execute("""
            ALTER TABLE graph_index_status
            ADD COLUMN IF NOT EXISTS existing_relations_count INTEGER DEFAULT 0;
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_feedback (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT NOW(),
                query TEXT,
                rating TEXT,
                missing_source TEXT,
                comment TEXT,
                metadata JSONB
            );
            """)

            conn.commit()
            cur.close()
            conn.close()

            print("DB initialized")
            return

        except psycopg2.OperationalError:
            print(f"DB not ready yet... retry {i}")
            time.sleep(2)

    raise Exception("DB not available after retries")
