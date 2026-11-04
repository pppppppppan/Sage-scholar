import { after, test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
mkdirSync(path.join(root, 'node_modules/.cache'), { recursive: true });
const output = mkdtempSync(path.join(root, 'node_modules/.cache/clarify-test-'));
after(() => rmSync(output, { recursive: true, force: true }));
for (const name of ['lib/api.ts', 'lib/conversations.ts', 'components/HumanInputCard.tsx']) {
  const source = readFileSync(path.join(root, name), 'utf8')
    .replace(/@\/lib\/([A-Za-z]+)/g, './$1.mjs').replace(/from "\.\/api"/g, 'from "./api.mjs"');
  writeFileSync(path.join(output, path.basename(name).replace(/\.tsx?$/, '.mjs')),
    ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 } }).outputText);
}
const { default: Card } = await import(pathToFileURL(path.join(output, 'HumanInputCard.mjs')));
const { answerRun } = await import(pathToFileURL(path.join(output, 'conversations.mjs')));
const run = { run_id: 'r', execution: { conversation_id: 'c' } };
const settings = { source_scope: 'public', allow_web: false, document_ids: [] };

test('permission clarification exposes explicit controls without pre-authorizing a change', () => {
  const html = renderToStaticMarkup(React.createElement(Card, { run, onChange() {}, question: {
    interrupt_id: 'i', question: 'Confirm scope', options: [], allow_skip: false,
    kind: 'source_policy', routing_settings: settings,
  } }));
  assert.match(html, /aria-label="资料范围"/);
  assert.match(html, /aria-label="联网权限"/);
  assert.match(html, /<option value="false" selected="">禁止联网/);
  assert.doesNotMatch(html, /type="checkbox"[^>]*checked/);
  assert.match(html, /<button disabled="">提交并继续/);
});

test('ordinary answer never converts permission words into configuration', async () => {
  const original = globalThis.fetch;
  const bodies = [];
  globalThis.fetch = async (_url, options) => {
    bodies.push(JSON.parse(options.body));
    return new Response(JSON.stringify({ run }), { status: 200 });
  };
  try {
    await answerRun(run, 'i', '允许联网');
    assert.equal(bodies[0].routing_update, undefined);
    await answerRun(run, 'i', '', false, { ...settings, allow_web: true });
    assert.deepEqual(bodies[1].routing_update, { ...settings, allow_web: true });
    assert.equal(bodies[1].interrupt_id, 'i');
    assert.equal(bodies[1].run_id, 'r');
    assert.equal(bodies[1].conversation_id, 'c');
  } finally { globalThis.fetch = original; }
});
