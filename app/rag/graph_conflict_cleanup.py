import argparse
from datetime import datetime, timezone

from app.rag.graph_store import RELATION_PRIORITY, driver


def relation_rank(row):
    relation = row.get("relation")
    sources = row.get("sources") or []

    return (
        RELATION_PRIORITY.get(relation, 0),
        min(len(sources), 25),
        -len(str(relation or "")),
        str(relation or ""),
    )


def unique_list(values):
    result = []
    seen = set()

    for value in values:
        if value is None:
            continue

        key = str(value)
        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def get_conflict_groups():
    query = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    WITH s, t, collect({
        rel_id: elementId(r),
        relation: r.type,
        sources: coalesce(r.sources, []),
        metadata: r.metadata
    }) AS rels
    WITH s, t, rels,
         size(apoc.coll.toSet([rel IN rels | rel.relation])) AS relation_type_count
    WHERE relation_type_count > 1
    RETURN s.name AS source,
           t.name AS target,
           relation_type_count,
           rels
    ORDER BY relation_type_count DESC, source, target
    """

    fallback_query = """
    MATCH (s:Entity)-[r:RELATED]->(t:Entity)
    RETURN s.name AS source,
           t.name AS target,
           elementId(r) AS rel_id,
           r.type AS relation,
           coalesce(r.sources, []) AS sources,
           r.metadata AS metadata
    ORDER BY source, target
    """

    with driver.session() as session:
        try:
            rows = [dict(row) for row in session.run(query)]
            return rows
        except Exception:
            pass

        grouped = {}
        for row in session.run(fallback_query):
            key = (row["source"], row["target"])
            grouped.setdefault(key, []).append({
                "rel_id": row["rel_id"],
                "relation": row["relation"],
                "sources": row["sources"],
                "metadata": row["metadata"],
            })

    conflicts = []
    for (source, target), rels in grouped.items():
        relation_types = {rel["relation"] for rel in rels}
        if len(relation_types) <= 1:
            continue
        conflicts.append({
            "source": source,
            "target": target,
            "relation_type_count": len(relation_types),
            "rels": rels,
        })

    conflicts.sort(key=lambda item: item["relation_type_count"], reverse=True)
    return conflicts


def choose_winner(rels):
    return sorted(rels, key=relation_rank, reverse=True)[0]


def merge_conflict_tx(tx, winner_id, loser_ids, merged_sources, merged_relation_types):
    query = """
    MATCH ()-[winner:RELATED]->()
    WHERE elementId(winner) = $winner_id
    SET winner.sources = $merged_sources,
        winner.merged_relation_types = $merged_relation_types,
        winner.merged_relation_count = size($merged_relation_types),
        winner.conflict_cleanup_at = $cleanup_at
    WITH winner
    MATCH ()-[loser:RELATED]->()
    WHERE elementId(loser) IN $loser_ids
    DELETE loser
    RETURN count(loser) AS deleted
    """
    row = tx.run(
        query,
        winner_id=winner_id,
        loser_ids=loser_ids,
        merged_sources=merged_sources,
        merged_relation_types=merged_relation_types,
        cleanup_at=datetime.now(timezone.utc).isoformat(),
    ).single()
    return int(row["deleted"]) if row else 0


def cleanup_conflicting_relations(dry_run=True, sample_limit=20, max_groups=None):
    groups = get_conflict_groups()
    if max_groups:
        groups = groups[:max(0, int(max_groups))]

    samples = []
    planned_deletes = 0
    applied_deletes = 0

    with driver.session() as session:
        for group in groups:
            rels = group["rels"]
            winner = choose_winner(rels)
            losers = [rel for rel in rels if rel["rel_id"] != winner["rel_id"]]
            merged_sources = unique_list(
                source
                for rel in rels
                for source in (rel.get("sources") or [])
            )
            merged_relation_types = unique_list(rel.get("relation") for rel in rels)
            planned_deletes += len(losers)

            if len(samples) < sample_limit:
                samples.append(
                    {
                        "source": group["source"],
                        "target": group["target"],
                        "keep": winner["relation"],
                        "merge_types": merged_relation_types,
                        "delete_relations": [rel["relation"] for rel in losers],
                        "source_count": len(merged_sources),
                    }
                )

            if not dry_run and losers:
                applied_deletes += session.execute_write(
                    merge_conflict_tx,
                    winner["rel_id"],
                    [rel["rel_id"] for rel in losers],
                    merged_sources,
                    merged_relation_types,
                )

    remaining_groups = len(get_conflict_groups()) if not dry_run else len(groups)

    return {
        "dry_run": dry_run,
        "conflict_groups": len(groups),
        "planned_deleted_relations": planned_deletes,
        "applied_deleted_relations": applied_deletes,
        "remaining_conflict_groups": remaining_groups,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()

    result = cleanup_conflicting_relations(
        dry_run=not args.apply,
        sample_limit=args.sample_limit,
        max_groups=args.max_groups,
    )

    for key, value in result.items():
        if key == "samples":
            print("samples:")
            for sample in value:
                print(
                    "  - "
                    f"{sample['source']} -> {sample['target']}: "
                    f"keep {sample['keep']}, merge {sample['merge_types']}, "
                    f"delete {sample['delete_relations']}, sources {sample['source_count']}"
                )
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
