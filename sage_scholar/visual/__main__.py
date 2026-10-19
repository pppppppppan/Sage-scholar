"""Rebuild uploaded PDFs without touching the existing text embedding index."""
import argparse
import json


def main():
    from sage_scholar.config import Config
    from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--document")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    retrieval = LegacyRetrievalAdapter(Config())
    docs = [d.document_id for d in retrieval.documents.list()
            if d.active_version and d.original_path and d.original_path.lower().endswith(".pdf")] if args.all else [args.document]
    failed = False
    for identity in docs:
        result = retrieval.visual.rebuild(identity)
        print(json.dumps({"document_id": identity, **result}, ensure_ascii=False))
        failed |= result["status"] == "failed"
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
