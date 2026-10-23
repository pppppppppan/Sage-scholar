"""Real production retrieval on a bounded, versioned evaluation corpus."""
from pathlib import Path
import hashlib
import json
from sage_scholar.config import Config
from sage_scholar.domain.documents import DocumentRecord, DocumentVersion, IngestionStatus
from sage_scholar.infrastructure.retrieval import LegacyRetrievalAdapter


class Corpus:
    def __init__(self, documents, root):
        key = hashlib.sha256(json.dumps(documents,sort_keys=True).encode()).hexdigest()
        self.directory = Path(root)/key
        self.directory.mkdir(parents=True,exist_ok=True)
        self.retrieval = LegacyRetrievalAdapter(Config(data_dir=str(self.directory)))
        self.by_id = {d['id']:d for d in documents}
        for item in documents:
            identity = item['id']
            name = hashlib.sha256(identity.encode()).hexdigest()+'.md'
            path = self.directory/name
            path.write_text(item['text'])
            version = item.get('version','v1')
            self.retrieval.pipeline.add_document(str(path))
            self.retrieval.documents.put(DocumentRecord(document_id=identity,title=item.get('title',identity),
                source_identifier=item.get('url','eval:'+identity),converted_path=str(path),
                active_version=version,document_version=version,parser_version='eval-source-checked-v1',
                ingestion_status=IngestionStatus.READY, versions=[DocumentVersion(document_version=version,
                    version_number=1,parser_version='eval-source-checked-v1',converted_path=str(path))]))

    def close(self):
        if self.retrieval._pipeline:
            self.retrieval._pipeline.vector_store.close()
