import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Offline Sage Scholar evaluation; live models require explicit flags')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run')
    run.add_argument('--suite',choices=['all','rag','router','supervisor','deep_research','citations','skills'],default='all')
    run.add_argument('--dataset')
    run.add_argument('--data-dir')
    run.add_argument('--output',default='eval-results')
    run.add_argument('--baseline')
    run.add_argument('--gate',help='Quality gate JSON; required not-evaluated is a nonzero exit')
    run.add_argument('--skill-ablation',action='store_true',help='Run Skills service twice, with/without Skill support; requires paid opt-in')
    run.add_argument('--allow-paid',action='store_true',help='Explicit second opt-in for Supervisor/Skills/Citation API execution')
    run.add_argument('--seed',type=int,default=42)
    run.add_argument('--k',type=int,default=5)
    run.add_argument('--answers',action='store_true',help='With --live --suite rag, explicitly call QA models to generate answers')
    run.add_argument('--live',action='store_true',help='Explicitly enable local Router or configured RAG models/APIs')
    run.add_argument('--judge-model',help='Explicit opt-in; requires EVAL_JUDGE_API_KEY')
    run.add_argument('--judge-base-url')
    bench=commands.add_parser('benchmark')
    bench.add_argument('--sizes',default='128,1024,4096')
    bench.add_argument('--seed',type=int,default=42)
    bench.add_argument('--queries',type=int,default=12)
    bench.add_argument('--output',default='eval-results')
    bench.add_argument('--models',choices=['fixture','local'],default='fixture')
    validate=commands.add_parser('validate')
    validate.add_argument('--dataset',required=True)
    compare=commands.add_parser('rerank')
    compare.add_argument('--dataset',required=True)
    compare.add_argument('--models',nargs='+',default=['cross-encoder/ms-marco-MiniLM-L-6-v2','BAAI/bge-reranker-v2-m3'])
    compare.add_argument('--device',default='cpu')
    compare.add_argument('--k',type=int,default=3)
    compare.add_argument('--output',default='eval-results')
    args=parser.parse_args()
    if args.command=='validate':
        from evals.dataset import load, is_labeled
        data=load(args.dataset)
        print({'samples':len(data),'labeled':sum(is_labeled(s) for s in data),'pending':sum(not is_labeled(s) for s in data)})
        return
    if args.command=='rerank':
        from evals.runners.rerank import compare
        compare(args)
        return
    if args.command=='run':
        if args.skill_ablation and not (args.live and args.allow_paid and args.suite=='skills'):
            parser.error('--skill-ablation requires --live --allow-paid --suite skills')
        if args.answers and (not args.live or args.suite != 'rag'):
            parser.error('--answers requires --live --suite rag')
        if args.k < 1:
            parser.error('--k must be positive')
        from evals.runners.runner import run as execute
        result=execute(args)
        if any('error' in sample for sample in result['samples']):
            raise SystemExit(1)
        if result['quality_gate']['status'] in {'failed','not_evaluated'}:
            raise SystemExit(2)
    else:
        from evals.runners.benchmark import benchmark
        benchmark(args)


if __name__=='__main__':
    main()
