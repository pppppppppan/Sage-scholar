from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import os
import shutil
import pytest
from mcp.types import Tool

from sage_scholar.mcp.lifecycle import MCPLifecycle
from sage_scholar.research.runtime import ResearchRuntimeProvider
from sage_scholar.skills import SkillService, SkillRegistry, SkillError
from sage_scholar.skills.service import STATE_KEY
from sage_scholar.runtime.journal import WorkflowInterrupted
from test_skills import setup, request
from test_document_evidence import library


class Client:
    def __init__(self, name):
        self.name, self.status, self.closed = name, 'available', 0
        names = {'pdfmux': ['convert_pdf'], 'paper-search': ['download_arxiv', 'read_arxiv_paper'], 'tavily': ['tavily_search']}.get(name, ['fetch'])
        self.tools = [Tool(name=n, description=n, inputSchema={'type': 'object'}) for n in names]
    def disconnect(self):
        self.closed += 1
        self.status = 'closed'
    def call_tool(self, name, args):
        return 'source'


def connector(fail_first, calls, clients):
    def connect(path, *, server_names, connect_timeout):
        calls.append(set(server_names))
        for name in sorted(server_names):
            if name in fail_first:
                fail_first.remove(name)
                yield {'server': name, 'status': 'unavailable', 'message': '不可用'}, None
            else:
                client = Client(name)
                clients.append(client)
                yield {'server': name, 'status': 'available', 'message': '已就绪'}, client
    return connect


def test_server_retry_reuses_healthy_connections_and_concurrent_adoption():
    clock, calls, clients = [0.], [], []
    pool = MCPLifecycle('unused', clock=lambda: clock[0])
    connect = connector({'fetch'}, calls, clients)
    pool.ensure({'fetch', 'paper-search'}, connect)
    paper = pool.registry.tools['mcp__paper-search__read_arxiv_paper']
    assert pool.states['fetch'].status == 'unavailable'
    pool.ensure({'fetch', 'paper-search'}, connect)
    assert len(calls) == 1
    clock[0] = 6
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(lambda _: pool.ensure({'fetch', 'paper-search'}, connect), range(8)))
    assert calls == [{'fetch', 'paper-search'}, {'fetch'}]
    assert pool.registry.tools['mcp__paper-search__read_arxiv_paper'] is paper
    assert len(pool.clients) == 2 and pool.states['fetch'].status == 'ready'
    pool.close(); pool.close()
    assert all(client.closed == 1 for client in clients)
    with pytest.raises(RuntimeError, match='closed'):
        pool.ensure({'fetch'}, connect)


def test_failed_library_initialization_is_not_cached_forever(library, monkeypatch):
    _, adapter, *_ = library
    provider = ResearchRuntimeProvider(adapter._config, lambda: pytest.fail('unneeded model'), adapter)
    clock, calls, clients = [0.], [], []
    provider._mcp.clock = lambda: clock[0]
    monkeypatch.setattr('sage_scholar.research.runtime.iter_mcp_clients', connector({'pdfmux', 'paper-search'}, calls, clients))
    for capability in ('pdf', 'paper'):
        with pytest.raises(RuntimeError, match='不可用'):
            provider.get_library_manager(capability)
    clock[0] = 6
    pdf = provider.get_library_manager('pdf').pdfmux_tool
    paper = provider.get_library_manager('paper').paper_tool
    assert pdf is provider.get_library_manager('pdf').pdfmux_tool
    assert paper is provider.get_library_manager('paper').paper_tool
    assert calls == [{'pdfmux'}, {'paper-search'}, {'pdfmux'}, {'paper-search'}]
    provider.close()
    assert all(c.closed == 1 for c in clients)


def test_failed_retry_has_bounded_backoff_and_no_spin():
    pool = MCPLifecycle('none', clock=lambda: 0.)
    calls = []
    def connect(path, **kwargs):
        calls.append(True)
        yield {'server': 'fetch', 'status': 'unavailable'}, None
    for _ in range(100):
        pool.ensure({'fetch'}, connect)
    assert len(calls) == 1 and pool.states['fetch'].retry_at > 0
    assert 'Traceback' not in str(pool.states['fetch'].public)


def test_cancelled_initialization_closes_generator_and_pool_retains_ownership(monkeypatch):
    pool = MCPLifecycle('none')
    client, closed = Client('fetch'), []
    def connect(path, **kwargs):
        try:
            yield {'server': 'fetch', 'status': 'available'}, client
            raise AssertionError('must not continue')
        finally:
            closed.append(True)
    checks = []
    def check():
        checks.append(True)
        if len(checks) == 2:
            raise WorkflowInterrupted('cancelled')
    monkeypatch.setattr('sage_scholar.mcp.lifecycle.check_control', check)
    with pytest.raises(WorkflowInterrupted):
        pool.ensure({'fetch', 'paper-search'}, connect)
    assert closed and pool.states['paper-search'].status == 'unavailable'
    pool.close()
    assert client.closed == 1


def test_skill_same_bytes_survive_mtime_inode_and_legal_root_move(setup, tmp_path):
    skills, _, store, root = setup
    req = request()
    old = skills.view(req, 'paper-review')
    path = root / 'paper-review/references/review-checklist.md'
    text = path.read_bytes()
    path.unlink(); path.write_bytes(text)
    os.utime(path, (1, 1))
    other = tmp_path / 'copy'
    shutil.copytree(root, other)
    restored = SkillService(store, SkillRegistry(other))
    assert restored.view(req, 'paper-review', 'references/review-checklist.md')['content'].encode() == text
    assert restored.view(request(run_id='new'), 'paper-review')['version'] == old['version']


def test_legacy_skill_state_and_corrupt_snapshot_fail_explicitly(setup):
    skills, _, store, _ = setup
    req = request()
    skills.view(req, 'paper-review')
    state = store.result(req.run_id, STATE_KEY)
    state['paper-review'].pop('schema_version')
    store.save_result(req.run_id, STATE_KEY, state)
    with pytest.raises(SkillError, match='旧 Skill'):
        skills.selected(req)
    state['paper-review']['schema_version'] = 2
    state['paper-review']['files']['SKILL.md'] += 'tamper'
    store.save_result(req.run_id, STATE_KEY, state)
    with pytest.raises(SkillError, match='校验失败'):
        skills.selected(req)


def test_oversize_reference_refused_before_snapshot(setup):
    skills, _, _, root = setup
    (root / 'paper-review/references/huge.md').write_text('x' * 32001)
    with pytest.raises(SkillError, match='过大'):
        skills.view(request(), 'paper-review')


def test_closed_library_connection_is_replaced_without_reusing_dead_adapter(library, monkeypatch):
    _, adapter, *_ = library
    clock, calls, clients = [0.], [], []
    provider = ResearchRuntimeProvider(adapter._config, lambda: None, adapter)
    provider._mcp.clock = lambda: clock[0]
    monkeypatch.setattr('sage_scholar.research.runtime.iter_mcp_clients', connector(set(), calls, clients))
    old = provider.get_library_manager('paper').paper_tool
    clients[0].status = 'unavailable'
    with pytest.raises(RuntimeError, match='不可用'):
        provider.get_library_manager('paper')
    clock[0] = 6
    new = provider.get_library_manager('paper').paper_tool
    assert new is not old and len(clients) == 2
    provider.close()
    assert all(c.closed == 1 for c in clients)


def test_broken_optional_registration_does_not_discard_healthy_server():
    pool = MCPLifecycle('none')
    bad, good = Client('fetch'), Client('paper-search')
    bad.tools = [object()]
    def connect(path, **kwargs):
        yield {'server': 'fetch', 'status': 'available'}, bad
        yield {'server': 'paper-search', 'status': 'available'}, good
    pool.ensure({'fetch', 'paper-search'}, connect)
    assert bad.closed == 1 and good.closed == 0
    assert pool.states['fetch'].status == 'unavailable' and pool.states['paper-search'].status == 'ready'
    assert 'mcp__paper-search__read_arxiv_paper' in pool.registry.tools
    pool.close()
