import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Offline Sage Scholar evaluation; live models require explicit flags')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run')
    run.add_argument('--suite',choices=['all','rag','router','deep_research','citations','skills'],default='all')
    run.add_argument('--dataset')
    run.add_argument('--data-dir')
    run.add_argument('--output',default='eval-results')
    run.add_argument('--baseline')
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
    args=parser.parse_args()
    if args.command=='run':
        if args.answers and (not args.live or args.suite != 'rag'):
            parser.error('--answers requires --live --suite rag')
        if args.k < 1:
            parser.error('--k must be positive')
        from evals.runners.runner import run as execute
        result=execute(args)
        if any('error' in sample for sample in result['samples']):
            raise SystemExit(1)
    else:
        from evals.runners.benchmark import benchmark
        benchmark(args)


if __name__=='__main__':
    main()
