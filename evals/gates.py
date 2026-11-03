"""Mode-aware quality gates: insufficient labels are not a pass."""
import statistics
from evals.metrics import classification


def metric(rows, name):
    if name.startswith('classification.'):
        label = 'workflow' if rows and rows[0].get('suite') == 'router' else 'status'
        pairs = [(r['sample']['expected'][label], r['observed'].get(label,'missing')) for r in rows
                 if label in r['sample'].get('expected',{}) and 'observed' in r and r.get('evaluated',True) and not r.get('error')]
        if not pairs:
            return None, 0
        report = classification(pairs)
        parts = name.split('.')
        if len(parts) == 3:
            values = report['per_label'].get(parts[1])
            return (values.get(parts[2]), values['support']) if values else (None,0)
        if parts[1].startswith('macro_'):
            key = parts[1][6:]
            values = [v[key] for v in report['per_label'].values() if v['support'] > 0]
            return statistics.mean(values), len(pairs)
        return report.get(parts[1]), len(pairs)
    values = [r['metrics'][name] for r in rows if name in r.get('metrics',{})]
    return (statistics.mean(values), len(values)) if values else (None,0)


def check(report, config, baseline=None):
    results = []
    for rule in config.get('rules', []):
        required = rule.get('required', True)
        modes = rule.get('modes', ['live','recorded_run','recorded_prediction'])
        rows = [r for r in report['samples'] if r['suite']==rule['suite'] and r.get('mode') in modes]
        value, count = metric(rows, rule['metric'])
        status, reason = 'passed', ''
        if any(r.get('error') for r in rows):
            status, reason = 'failed', 'Execution errors in evaluated samples'
        elif value is None or count < rule.get('min_samples', 10):
            status, reason = 'not_evaluated', 'Missing gold/metric or insufficient labeled samples'
        elif ('min' in rule and value < rule['min']) or ('max' in rule and value > rule['max']):
            status, reason = 'failed', 'Absolute threshold violated'
        delta = None
        if status == 'passed' and ('min_delta' in rule or 'max_delta' in rule):
            if not baseline or not report.get('baseline_compatible'):
                status, reason = 'not_evaluated', 'Missing/incompatible baseline'
            else:
                before = [r for r in baseline['samples'] if r['suite']==rule['suite'] and r.get('mode') in modes]
                reference, n = metric(before, rule['metric'])
                if reference is None or n < rule.get('min_samples',10):
                    status, reason = 'not_evaluated', 'Insufficient baseline labels'
                else:
                    delta = value-reference
                    if ('min_delta' in rule and delta < rule['min_delta']) or ('max_delta' in rule and delta > rule['max_delta']):
                        status, reason = 'failed', 'Baseline threshold violated'
        results.append({**rule,'required':required,'value':value,'n':count,'delta':delta,'status':status,'reason':reason})
    required = [r for r in results if r['required']]
    state = ('failed' if any(r['status']=='failed' for r in required) else
             'not_evaluated' if not required or any(r['status']=='not_evaluated' for r in required) else 'passed')
    return {'status':state,'rules':results}
