"""Serialization of small research messages; originals stay in EvidenceStore."""
import json


def encoded(value):
    return json.dumps(value, ensure_ascii=False)
