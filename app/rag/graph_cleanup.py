import argparse

from app.rag.graph_extract import normalize_entity, normalize_relation
from app.rag.graph_schema import ALLOWED_RELATIONS
from app.rag.graph_store import driver


def get_relations():
    query = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    RETURN elementId(r) AS rel_id,
           s.name AS source,
           r.type AS relation,
           t.name AS target,
           coalesce(r.sources, []) AS sources,
           r.metadata AS metadata
    """

    with driver.session() as session:
        return [dict(row) for row in session.run(query)]


def delete_relation(tx, rel_id):
    query = """
    MATCH ()-[r:RELATED]->()
    WHERE elementId(r) = $rel_id
    DELETE r
    """
    tx.run(query, rel_id=rel_id)


def rewrite_relation(tx, rel_id, source, relation, target):
    query = """
    MATCH (old_s:Entity)-[old:RELATED]->(old_t:Entity)
    WHERE elementId(old) = $rel_id
    MERGE (s:Entity {name: $source})
    MERGE (t:Entity {name: $target})
    MERGE (s)-[r:RELATED {type: $relation}]->(t)
    SET r.metadata = coalesce(old.metadata, r.metadata),
        r.sources = reduce(
            acc = coalesce(r.sources, []),
            item IN coalesce(old.sources, []) |
            CASE WHEN item IN acc THEN acc ELSE acc + item END
        )
    WITH old, r
    WHERE elementId(old) <> elementId(r)
    DELETE old
    """
    tx.run(
        query,
        rel_id=rel_id,
        source=source,
        relation=relation,
        target=target
    )


def delete_isolated_nodes(tx):
    query = """
    MATCH (n:Entity)
    WHERE NOT (n)--()
    DELETE n
    RETURN count(n) AS deleted
    """
    row = tx.run(query).single()
    return int(row["deleted"]) if row else 0


def relation_type_counts():
    query = """
    MATCH ()-[r:RELATED]->()
    RETURN r.type AS relation, count(*) AS count
    ORDER BY count DESC, relation
    """

    with driver.session() as session:
        return [dict(row) for row in session.run(query)]


def cleanup_graph(dry_run=True, sample_limit=20, strict_relations=True):
    rows = get_relations()
    keep = []
    delete = []
    rewrite = []
    invalid_relation_counts = [
        row
        for row in relation_type_counts()
        if row["relation"] not in ALLOWED_RELATIONS
    ]

    for row in rows:
        raw = {
            "source": row["source"],
            "relation": row["relation"],
            "target": row["target"]
        }
        cleaned_source = normalize_entity(raw["source"])
        cleaned_target = normalize_entity(raw["target"])
        cleaned_relation = normalize_relation(raw["relation"])

        if not cleaned_source or not cleaned_target or cleaned_source == cleaned_target:
            delete.append(row)
            continue

        if not cleaned_relation:
            if strict_relations:
                delete.append(row)
                continue

            cleaned_relation = raw["relation"]

        if (
            cleaned_source != row["source"]
            or cleaned_relation != row["relation"]
            or cleaned_target != row["target"]
        ):
            rewrite.append({
                **row,
                "clean_source": cleaned_source,
                "clean_relation": cleaned_relation,
                "clean_target": cleaned_target
            })
            continue

        keep.append(row)

    isolated_deleted = 0

    if not dry_run:
        with driver.session() as session:
            for row in delete:
                session.execute_write(delete_relation, row["rel_id"])

            for row in rewrite:
                session.execute_write(
                    rewrite_relation,
                    row["rel_id"],
                    row["clean_source"],
                    row["clean_relation"],
                    row["clean_target"]
                )

            isolated_deleted = session.execute_write(delete_isolated_nodes)

    return {
        "dry_run": dry_run,
        "total_relations": len(rows),
        "keep_relations": len(keep),
        "delete_relations": len(delete),
        "rewrite_relations": len(rewrite),
        "isolated_nodes_deleted": isolated_deleted,
        "invalid_relation_types": len(invalid_relation_counts),
        "invalid_relation_samples": [
            f"{r['relation']}: {r['count']}"
            for r in invalid_relation_counts[:sample_limit]
        ],
        "delete_samples": [
            f"{r['source']} --{r['relation']}--> {r['target']}"
            for r in delete[:sample_limit]
        ],
        "rewrite_samples": [
            (
                f"{r['source']} --{r['relation']}--> {r['target']}"
                f" => {r['clean_source']} --{r['clean_relation']}--> {r['clean_target']}"
            )
            for r in rewrite[:sample_limit]
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--strict-relations", action="store_true", default=True)
    parser.add_argument("--keep-unknown-relations", action="store_false", dest="strict_relations")
    parser.add_argument("--sample-limit", type=int, default=20)
    args = parser.parse_args()

    result = cleanup_graph(
        dry_run=not args.apply,
        sample_limit=args.sample_limit,
        strict_relations=args.strict_relations
    )

    for key, value in result.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
