"""Embedded KuzuDB backend: the production graph store (Cypher, no Docker).

Selected only when GRAPH_PATH points at a directory (not ":memory:"), and
imported lazily by store._backend so the keyless / in-memory path never needs
the wheel. Same interface as InMemoryGraph.

Schema: Entity{name PK, type} + REL{predicate}. Traversal is undirected
(matched with `-[...]-`), so two projects sharing a technology are connected
through it, mirroring the in-memory seam. max_hops bounds the variable-length
match so cyclic relationships can't run away.
"""
from __future__ import annotations

from .extract import Triple
from .store import GraphPath, Step


class KuzuGraph:
    def __init__(self, path: str) -> None:
        import kuzu

        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity(name STRING, type STRING, PRIMARY KEY(name))"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS REL(FROM Entity TO Entity, predicate STRING)"
        )

    def reset(self) -> None:
        # Drop the data by recreating the tables.
        self._conn.execute("DROP TABLE IF EXISTS REL")
        self._conn.execute("DROP TABLE IF EXISTS Entity")
        self._ensure_schema()

    def add_triples(self, triples: list[Triple]) -> None:
        for t in triples:
            self._conn.execute(
                "MERGE (e:Entity {name: $n}) SET e.type = $ty",
                {"n": t.subj, "ty": t.subj_type},
            )
            self._conn.execute(
                "MERGE (e:Entity {name: $n}) SET e.type = $ty",
                {"n": t.obj, "ty": t.obj_type},
            )
            self._conn.execute(
                "MATCH (a:Entity {name: $a}), (b:Entity {name: $b}) "
                "MERGE (a)-[r:REL {predicate: $p}]->(b)",
                {"a": t.subj, "b": t.obj, "p": t.pred},
            )

    def has_node(self, name: str) -> bool:
        res = self._conn.execute(
            "MATCH (e:Entity {name: $n}) RETURN count(e)", {"n": name}
        )
        return bool(res.get_next()[0]) if res.has_next() else False

    def node_names(self) -> set[str]:
        res = self._conn.execute("MATCH (e:Entity) RETURN e.name")
        out: set[str] = set()
        while res.has_next():
            out.add(res.get_next()[0])
        return out

    def node_type(self, name: str) -> str | None:
        res = self._conn.execute(
            "MATCH (e:Entity {name: $n}) RETURN e.type", {"n": name}
        )
        return res.get_next()[0] if res.has_next() else None

    def neighbourhood(self, name: str, hops: int = 1) -> list[GraphPath]:
        if not self.has_node(name) or hops < 1:
            return []
        res = self._conn.execute(
            "MATCH (a:Entity {name: $n})-[r:REL]-(b:Entity) "
            "RETURN a.name, r.predicate, b.name, "
            "startNode(r).name = a.name AS forward",
            {"n": name},
        )
        out: list[GraphPath] = []
        while res.has_next():
            src, pred, dst, forward = res.get_next()
            out.append(GraphPath((Step(src, pred, dst, bool(forward)),)))
        return out

    def shortest_paths(self, a: str, b: str, max_hops: int) -> list[GraphPath]:
        if not self.has_node(a) or not self.has_node(b) or a == b:
            return []
        res = self._conn.execute(
            f"MATCH p = (x:Entity {{name: $a}})-[:REL* SHORTEST 1..{max_hops}]-"
            "(y:Entity {name: $b}) RETURN nodes(p), rels(p)",
            {"a": a, "b": b},
        )
        out: list[GraphPath] = []
        while res.has_next():
            nodes, rels = res.get_next()
            steps: list[Step] = []
            for i, rel in enumerate(rels):
                src_name = nodes[i]["name"]
                dst_name = nodes[i + 1]["name"]
                forward = rel["_src"]["offset"] == nodes[i]["_id"]["offset"]
                steps.append(Step(src_name, rel["predicate"], dst_name, forward))
            out.append(GraphPath(tuple(steps)))
        return out

    def stats(self) -> dict:
        n = self._conn.execute("MATCH (e:Entity) RETURN count(e)")
        e = self._conn.execute("MATCH ()-[r:REL]->() RETURN count(r)")
        nodes = n.get_next()[0] if n.has_next() else 0
        edges = e.get_next()[0] if e.has_next() else 0
        return {"nodes": nodes, "edges": edges}
