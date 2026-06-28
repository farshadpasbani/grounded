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

    sub.add_parser("eval", help="score retrieval + groundedness on the labelled set")

    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest import build

        stats = build()
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
        from .evaluate import run

        run()


if __name__ == "__main__":
    main()
