"""Explicit migration/recovery without loading an embedding model or parsing PDFs."""
import argparse
from pathlib import Path
from .persistent_store import open_index
from sage_scholar.infrastructure.document_store import JsonDocumentStore
from sage_scholar.infrastructure.json_store import file_lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True)
    args=parser.parse_args()
    directory=Path(args.data_dir)
    with file_lock(directory/'.ingestion'):
        store=open_index(directory/'rag.sqlite3',directory/'vector_store.json')
        try:
            documents=JsonDocumentStore(str(directory)).list()
            removed=store.reconcile_library(directory,documents,save=store.save)
            store.bind_catalog(documents)
            print(f'RAG schema 3: {len(store.chunks)} chunks, revision {store.index_revision}, removed {len(removed)} orphan paths. Legacy JSON retained unchanged.')
        finally:
            store.close()


if __name__=='__main__':
    main()
