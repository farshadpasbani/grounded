"""grounded ingest | ask "..." | eval"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="grounded")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="build the index from sources.yaml")

    ask_p = sub.add_parser("ask", help="ask a question")
    ask_p.add_argument("question")

    eval_p = sub.add_parser("eval", help="score retrieval + groundedness on the labelled set")
    eval_p.add_argument(
        "--graph",
        action="store_true",
        help="score the knowledge graph (graph-recall + groundedness) instead",
    )

    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest import build

        stats = build()
        if "nodes" in stats:  # graph / hybrid build
            print(f"built graph: {stats['nodes']} nodes, {stats['edges']} edges "
                  f"from {stats['files']} files")
        else:  # vector build
            print(f"indexed {stats['chunks']} chunks from {stats['files']} files "
                  f"({stats['ip_dropped']} dropped by ip-guard)")

    elif args.cmd == "ask":
        from .query import ask

        a = ask(args.question)
        print(a.text)
        if a.grounded:
            print("\nsources:")
            for h in a.citations:
                print(f"  [{h.id}] {h.source} :: {h.heading or 'intro'}  ({h.score:.2f})")
        else:
            print(f"\n(abstained: {a.reason})", file=sys.stderr)

    elif args.cmd == "eval":
        from .evaluate import run, run_graph

        run_graph() if args.graph else run()


if __name__ == "__main__":
    main()
