"""Actual local Router + persistent entry graph; answer services are stubs, no cloud calls."""
import asyncio, json, tempfile
from pathlib import Path
from types import SimpleNamespace
from test_semantic_query_graph import setup
from test_clarify_resume import Research, response
from sage_scholar.routing.classifier import LocalClassifier
from sage_scholar.application.query_service import QueryCommand
from sage_scholar.runtime.human import RoutingSettings

class Docs:
    async def catalog(self):
        return [{'document_id': 'a', 'title': 'Adaptive Mesh Refinement', 'ready': True},
                {'document_id': 'b', 'title': 'Neural Operators', 'ready': True},
                {'document_id': 'c', 'title': 'Fluid Simulation', 'ready': True}]
    async def resolve(self, query, ids):
        return SimpleNamespace(message=None, document_ids=ids)

async def main():
    rows=[]
    cases=[('missing_topic', '帮我调研一下', {}, {'answer':'自适应网格方法的研究进展'}, 'deep_research'),
           ('ambiguous_papers', '比较这两篇的方法', {}, {'routing_update':RoutingSettings(document_ids=['a','b'])}, 'qa'),
           ('web_permission', '调研自适应网格方法的最新研究', {'allow_web':False}, {'routing_update':RoutingSettings(allow_web=True)}, 'deep_research')]
    for name,query,options,reply,expected in cases:
        with tempfile.TemporaryDirectory(prefix='router-resume-live-') as directory:
            research=Research();c=setup(Path(directory),LocalClassifier(),research,docs=Docs())
            try:
                conversation=c.store.conversations.create().conversation_id
                initial=await c.service.execute(QueryCommand(query=query,conversation_id=conversation,**options))
                await c.manager._tasks[initial.run_id]
                run=c.manager.get(initial.run_id)
                row={'case':name,'query':query,'initial_status':str(run.status),'initial_workflow':str(run.workflow),'interrupts':run.interrupts,'expected_workflow':expected}
                if run.status=='waiting_for_user':
                    await c.manager.answer(run.run_id,response(run,**reply))
                    await c.manager._tasks[run.run_id]
                final=c.manager.get(initial.run_id)
                row.update(final_status=str(final.status),final_workflow=str(final.workflow),error=final.error,
                           resumed_query=final.execution.query,final_interrupts=final.interrupts,
                           passed=run.status=='waiting_for_user' and final.status=='completed' and final.workflow==expected)
                rows.append(row)
            finally:await c.close()
    target=Path('eval-results/router-clarify/live-resume.json');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'mode':'live local Router + real SQLite/LangGraph + stub answer/research services','cases':rows},ensure_ascii=False,indent=2)+'\n')
    print([(r['case'],r['passed'],r['final_workflow']) for r in rows])
    if not all(r['passed'] for r in rows):raise SystemExit(1)
asyncio.run(main())
