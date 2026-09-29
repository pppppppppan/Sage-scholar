import { after, test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

// Test the actual TSX components with React SSR, without another test dependency.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cache = path.join(root, "node_modules/.cache");
mkdirSync(cache, { recursive: true });
const output = mkdtempSync(path.join(cache, "sage-markdown-test-"));
after(() => rmSync(output, { recursive: true, force: true }));
const sources = ["lib/markdown.ts", "lib/api.ts", "components/MarkdownContent.tsx", "components/InlineAnswer.tsx", "components/ReportView.tsx"];
for (const name of sources) {
  const source = readFileSync(path.join(root, name), "utf8").replace(/@\/(?:components|lib)\/([A-Za-z]+)/g, "./$1.mjs");
  const compiled = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022,
  }}).outputText;
  writeFileSync(path.join(output, path.basename(name).replace(/\.tsx?$/, ".mjs")), compiled);
}
const { default: Markdown } = await import(pathToFileURL(path.join(output, "MarkdownContent.mjs")));
const { default: InlineAnswer } = await import(pathToFileURL(path.join(output, "InlineAnswer.mjs")));
const { default: ReportView } = await import(pathToFileURL(path.join(output, "ReportView.mjs")));
const render = (text) => renderToStaticMarkup(React.createElement(Markdown, null, text));
const inline = String.raw`$D_{KL}(P\|Q)$`;
const block = String.raw`$$
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$`;

for (const [name, component, props] of [
  ["Direct", InlineAnswer, { answer: inline + "\n\n" + block, evidence: [] }],
  ["QA", InlineAnswer, { answer: inline + "\n\n" + block, evidence: [{ evidence_id: "ev_a", source_id: "paper", metadata: {}, section: null, page: null, text: "source" }] }],
  ["Report", ReportView, { report: inline + "\n\n" + block, stats: null }],
]) {
  test(`${name} renders inline and display math through the shared component`, () => {
    const html = renderToStaticMarkup(React.createElement(component, props));
    assert.match(html, /class="katex"/);
    assert.match(html, /class="katex-display"/);
    assert.doesNotMatch(html, /katex-error/);
  });
}

test("same-line display math is normalized", () => {
  assert.match(render("$$x^2+y^2=z^2$$"), /katex-display/);
});
test("ordinary money and shell variables remain literal, even together", () => {
  const html = render("价格 $10 和 $20；变量 $PATH 和 $HOME。另有 $x + y$。");
  assert.match(html, /\$10 和 \$20/);
  assert.match(html, /\$PATH 和 \$HOME/);
  assert.equal((html.match(/class="katex"/g) || []).length, 1);
});
test("escaped currency and code examples are not normalized as math", () => {
  const html = render("\\$10\n\n`$PATH`\n\n```sh\necho $PATH\n$$x$$\n```");
  assert.match(html, /\$10/);
  assert.match(html, /echo \$PATH/);
  assert.doesNotMatch(html, /class="katex/);
});
test("GFM tables and legacy report source links remain usable", () => {
  assert.match(render("| A | B |\n|---|---|\n| 1 | 2 |"), /<table>/);
  const html = renderToStaticMarkup(React.createElement(ReportView, { report: "Result [1]\n\n## Sources\n[1] https://example.org", stats: null }));
  assert.match(html, /href="#source-1"/);
  assert.match(html, /id="source-1"/);
});
test("both views use one renderer and KaTeX styles load globally", () => {
  for (const name of ["InlineAnswer", "ReportView"]) {
    const source = readFileSync(path.join(root, `components/${name}.tsx`), "utf8");
    assert.match(source, /import MarkdownContent/);
    assert.doesNotMatch(source, /import ReactMarkdown|function sanitizeMathDollars/);
  }
  assert.match(readFileSync(path.join(root, "app/layout.tsx"), "utf8"), /katex\/dist\/katex.min.css/);
});

test("SSE decoder forwards tool initialization progress across chunk boundaries", async (t) => {
  const { startRunEvents } = await import(pathToFileURL(path.join(output, "api.mjs")));
  const events = [
    { type: "initializing_tools", message: "正在初始化研究工具…" },
    { type: "tool_available", server: "paper-search", message: "论文搜索与读取已就绪" },
    { type: "tool_unavailable", server: "fetch", reason: "timeout", message: "网页读取不可用，将继续使用其他来源" },
    { type: "research_started", message: "正在规划研究任务…" },
  ];
  const wire = new TextEncoder().encode(events.map(({ type, ...data }) => `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`).join(""));
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({
    start(controller) {
      for (let i = 0; i < wire.length; i += 13) controller.enqueue(wire.slice(i, i + 13));
      controller.close();
    },
  })));
  const received = [];
  const done = Promise.withResolvers();
  const timeout = setTimeout(() => done.reject(new Error("SSE events not delivered")), 1000);
  const cancel = startRunEvents("test-run", (event) => {
    received.push(event);
    if (received.length === events.length) done.resolve();
  });
  try {
    await done.promise;
    assert.deepEqual(received, events);
  } finally {
    clearTimeout(timeout);
    cancel();
  }
});

test("resumed SSE skips old terminal events but replays results and current completion", async (t) => {
  const { startRunEvents } = await import(pathToFileURL(path.join(output, "api.mjs")));
  const wire = [
    [1, "run_interrupted", {}], [2, "run_resumed", {}],
    [3, "write", { report: "resumed report" }], [4, "run_finished", { status: "completed" }],
  ].map(([id, type, data]) => `id: ${id}\nevent: ${type}\ndata: ${JSON.stringify(data)}\n\n`).join("");
  t.mock.method(globalThis, "fetch", async () => new Response(wire));
  const received = [], done = Promise.withResolvers();
  const timeout = setTimeout(() => done.reject(new Error("no completion")), 1000);
  const cancel = startRunEvents("resumed", event => { received.push(event); if (event.type === "run_finished") done.resolve(); }, 2);
  try {
    await done.promise;
    assert.deepEqual(received.map(e => e.type), ["run_resumed", "write", "run_finished"]);
  } finally { clearTimeout(timeout); cancel(); }
});
