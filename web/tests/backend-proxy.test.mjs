import { test, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const directory = mkdtempSync(path.join(tmpdir(), "sage-proxy-"));
const source = readFileSync(new URL("../lib/backend-proxy.ts", import.meta.url), "utf8");
writeFileSync(path.join(directory, "proxy.mjs"), ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText);
const { proxyBackend } = await import(pathToFileURL(path.join(directory, "proxy.mjs")));
after(() => rmSync(directory, { recursive: true, force: true }));

async function fixture(fn) {
  const originalFetch = globalThis.fetch, originalUrl = process.env.BACKEND_INTERNAL_URL;
  process.env.BACKEND_INTERNAL_URL = "http://backend:8000";
  try { await fn(); } finally {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.BACKEND_INTERNAL_URL;
    else process.env.BACKEND_INTERNAL_URL = originalUrl;
  }
}

test("SSE forwards cursor and streams before upstream closes; cancel propagates", () => fixture(async () => {
  let controller, cancelled = false;
  const request = new Request("http://localhost:3000/api/runs/r/events?after=3", { headers: { "Last-Event-ID": "3" } });
  globalThis.fetch = async (url, options) => {
    assert.equal(url.href, "http://backend:8000/api/runs/r/events?after=3");
    assert.equal(options.headers.get("last-event-id"), "3");
    assert.equal(options.signal, request.signal);
    return new Response(new ReadableStream({ start(c) { controller = c; }, cancel() { cancelled = true; } }), {
      headers: { "Content-Type": "text/event-stream", "Content-Encoding": "gzip", "Content-Length": "999" },
    });
  };
  const result = await proxyBackend(request);
  assert.equal(result.headers.get("content-length"), null);
  assert.equal(result.headers.get("content-encoding"), null);
  const reader = result.body.getReader();
  controller.enqueue(new TextEncoder().encode("id: 4\nevent: answer_delta\ndata: {}\n\n"));
  assert.match(new TextDecoder().decode((await reader.read()).value), /answer_delta/);
  await reader.cancel(); assert.equal(cancelled, true);
}));

test("PDF multipart uploads remain streamed and HTTP validation errors preserved", () => fixture(async () => {
  const form = new FormData(); form.append("file", new Blob(["PDF-data"]), "test.pdf");
  globalThis.fetch = async (_url, options) => {
    assert.equal(options.duplex, "half");
    assert.match(options.headers.get("content-type"), /multipart\/form-data; boundary=/);
    assert.match(await new Response(options.body).text(), /PDF-data/);
    return Response.json({ detail: "Invalid PDF" }, { status: 422 });
  };
  const result = await proxyBackend(new Request("http://local/api/library/upload", { method: "POST", body: form }));
  assert.equal(result.status, 422); assert.equal((await result.json()).detail, "Invalid PDF");
}));

test("redirects stay same-origin and incoming host cannot change backend", () => fixture(async () => {
  globalThis.fetch = async (url) => {
    assert.equal(url.host, "backend:8000");
    return new Response(null, { status: 307, headers: { location: "http://backend:8000/api/library/" } });
  };
  const result = await proxyBackend(new Request("http://evil.example/api/library"));
  assert.equal(result.headers.get("location"), "/api/library/");
}));

test("unconfigured and unavailable backend fail clearly without internal errors", () => fixture(async () => {
  delete process.env.BACKEND_INTERNAL_URL;
  assert.equal((await proxyBackend(new Request("http://local/api/health"))).status, 503);
  process.env.BACKEND_INTERNAL_URL = "http://backend:8000";
  globalThis.fetch = async () => { throw new Error("private internal details"); };
  const response = await proxyBackend(new Request("http://local/api/health"));
  assert.equal(response.status, 502); assert.doesNotMatch(await response.text(), /private/);
}));
