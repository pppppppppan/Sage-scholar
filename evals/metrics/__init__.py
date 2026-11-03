"""Metrics only consume declared gold/annotations; absence means unassessed."""
import math
from collections import Counter


def retrieval_metrics(retrieved, gold, k=5):
    if gold is None:
        return {}
    ranked = list(dict.fromkeys(retrieved))[:k]
    relevance = gold if isinstance(gold, dict) else {x: 1 for x in gold}
    relevant = {x for x, grade in relevance.items() if grade > 0}
    if not relevant:
        return {}  # No relevant labels is not a meaningful recall denominator.
    hits = [int(x in relevant) for x in ranked]
    dcg = sum((2 ** relevance.get(x, 0) - 1) / math.log2(i+2) for i, x in enumerate(ranked))
    ideal = sum((2 ** grade - 1)/math.log2(i+2) for i, grade in enumerate(sorted(relevance.values(),reverse=True)[:k]))
    return {f'recall@{k}': sum(hits)/len(relevant), f'precision@{k}': sum(hits)/k,
            'mrr': next((1/(i+1) for i, hit in enumerate(hits) if hit), 0), 'ndcg': dcg/ideal if ideal else 0}


def classification(pairs):
    labels = sorted({label for pair in pairs for label in pair})
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    for gold, pred in pairs:
        matrix[gold][pred] += 1
    per_label = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in labels if g != label)
        fn = sum(matrix[label][p] for p in labels if p != label)
        precision, recall = tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0
        per_label[label] = {'precision': precision, 'recall': recall,
                            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
                            'support': tp+fn}
    return {'accuracy': sum(a==b for a,b in pairs)/len(pairs) if pairs else None,
            'confusion_matrix': matrix, 'per_label': per_label}


def annotations(observed):
    """Only explicit human labels produce these semantic scores, never regex proxies."""
    result = {}
    for name, rows in observed.get('annotations', {}).items():
        if name not in {'answer_correctness','faithfulness','citation_accuracy','task_coverage',
                        'task_redundancy','evidence_quality','note_quality','report_quality'}:
            continue
        if not isinstance(rows, list):
            raise ValueError('annotations must be lists of scored units')
        for row in rows:
            if row.get('method') != 'human' or not row.get('rationale') or not row.get('annotator'):
                raise ValueError('semantic annotations require human annotator and rationale')
            if not isinstance(row.get('score'), (float,int)) or not 0 <= row['score'] <= 1:
                raise ValueError('annotation score outside [0,1]')
        if rows:
            result[name] = sum(r['score'] for r in rows)/len(rows)
    return result


def evaluate(suite, sample, observed, k):
    expected = sample.get('expected', {})
    scores = annotations(observed)
    if suite == 'rag':
        scores.update(retrieval_metrics(observed.get('retrieved', []), sample.get('gold_evidence'), k))
        # Exact match is a separately named rule, not semantic correctness.
        if 'answer' in expected and 'answer' in observed:
            scores['answer_exact_match'] = float(observed['answer'].strip() == expected['answer'].strip())
        if 'citations' in expected:
            scores['citation_identity_precision'] = len(set(observed.get('citations', [])) & set(expected['citations'])) / max(1,len(set(observed.get('citations', []))))
    elif suite == 'router':
        for key in ('workflow','source_scope','allow_web','document_ids','referent'):
            if key in expected:
                scores[key+'_accuracy'] = float(expected[key] == observed.get(key))
        fields = [key for key in ('source_scope','allow_web','document_ids') if key in expected]
        if {'source_scope','allow_web'} <= set(fields):
            scores['source_policy_accuracy'] = float(all(expected[k] == observed.get(k) for k in fields))
        from evals.runners.router_diagnostics import violations
        scores['permission_violations'] = len(violations(sample,observed))
        if 'workflow' in expected and expected['workflow'] != 'clarify':
            scores['false_clarify_rate'] = float(observed.get('workflow') == 'clarify')
        if 'unavailable' in expected:
            scores['unavailable_handling'] = float(expected['unavailable'] == observed.get('unavailable', False))
    elif suite == 'citations':
        if 'status' in expected:
            scores['status_accuracy'] = float(expected['status'] == observed.get('status'))
    elif suite == 'skills':
        if 'selected_skills' in expected:
            scores['selection_exact_match'] = float(set(expected['selected_skills']) == set(observed.get('selected_skills', [])))
        if 'required_source_tools' in expected and observed.get('source_tool_trace_complete',True):
            required = set(expected['required_source_tools'])
            scores['required_tool_coverage'] = len(required & set(observed.get('source_tools', [])))/max(1,len(required))
        pair = observed.get('paired_quality')
        if pair:
            if pair.get('method') != 'human' or not pair.get('rationale') or not pair.get('same_input'):
                raise ValueError('Skill comparison requires an annotated same-input pair')
            scores['paired_quality_delta'] = pair['with_skill'] - pair['without_skill']
    elif suite in {'supervisor','deep_research'}:
        tasks = observed.get('tasks', [])
        notes = observed.get('notes', [])
        scores['task_count'] = len(tasks)
        if 'max_tasks' in expected:
            scores['task_count_within_gold_limit'] = float(expected.get('min_tasks',1) <= len(tasks) <= expected['max_tasks'])
        if tasks:
            scores['exact_duplicate_task_rate'] = 1-len({t.get('objective','').strip() for t in tasks})/len(tasks)
        evidence_ids = {e['evidence_id'] for e in observed.get('evidence', [])}
        references = [s for n in notes for s in n.get('source_ids', [])]
        if references:
            scores['note_source_resolvable_rate'] = sum(s in evidence_ids for s in references)/len(references)
        # Neither resolvability nor self-reported covers proves scientific support.
    for key in ('latency_ms','retrieval_ms','prompt_tokens','completion_tokens','tool_calls','active_seconds','cost','skill_overhead_tokens','skill_overhead_ms'):
        if key in observed and observed[key] is not None:
            scores[key] = observed[key]
    pricing = sample.get('pricing_per_million')
    if pricing and {'prompt_tokens','completion_tokens'} <= observed.keys():
        scores['estimated_cost'] = (observed['prompt_tokens']*pricing['input'] + observed['completion_tokens']*pricing['output'])/1e6
    return scores
