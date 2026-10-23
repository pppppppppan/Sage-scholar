"""Dataset validation; pending annotation never becomes a quality score."""
import hashlib
import json
from pathlib import Path


def load(path):
    path = Path(path)
    samples = []
    seen = set()
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ('id', 'query', 'dataset_version', 'provenance'):
            if not item.get(key):
                raise ValueError(f'{path}:{index}: missing {key}')
        if item['id'] in seen or item['id'] in {'.','..'} or Path(item['id']).name != item['id']:
            raise ValueError(f'{path}:{index}: duplicate or invalid ID')
        seen.add(item['id'])
        if item.get('corpus'):
            corpus_path = (path.parent / item['corpus']).resolve()
            payload = corpus_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != item.get('corpus_sha256'):
                raise ValueError(f'{item["id"]}: corpus hash mismatch; review labels before updating')
            item['documents'] = json.loads(payload)
        annotation = item.get('annotation')
        if annotation and annotation.get('status') == 'verified':
            if not all(annotation.get(key) for key in ('reviewer','method','basis')):
                raise ValueError('Verified gold requires reviewer/method/basis')
            if item['provenance'].get('kind') == 'real_source' and not item.get('sources'):
                raise ValueError('Real-source gold requires versioned source locators')
        if item.get('documents'):
            ids = [d['id'] for d in item['documents']]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate corpus document IDs')
            # Corpus tests rank stable passage IDs; live-library tests use Evidence IDs.
            if item.get('gold_evidence') and not set(item['gold_evidence']) <= set(ids):
                raise ValueError('Gold passage not found in corpus')
        samples.append(item)
    if not samples:
        raise ValueError('Empty dataset')
    return samples


def is_labeled(sample):
    annotation = sample.get('annotation')
    if annotation:
        return annotation.get('status') == 'verified'
    return sample.get('provenance', {}).get('kind') != 'real_source'
