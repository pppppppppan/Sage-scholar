/** Streaming same-origin bridge; backend URL is server-only and read at runtime. */
export async function proxyBackend(request: Request): Promise<Response> {
  const origin = process.env.BACKEND_INTERNAL_URL;
  if (!origin) return Response.json({ detail: "Backend proxy is not configured" }, { status: 503 });
  const incoming = new URL(request.url);
  if (!incoming.pathname.startsWith("/api/")) return new Response(null, { status: 404 });
  const target = new URL(origin);
  // Assign pathname instead of resolving user-supplied // paths against the origin.
  target.pathname = incoming.pathname;
  target.search = incoming.search;
  const headers = new Headers();
  for (const name of ["accept", "content-type", "last-event-id", "range", "if-range"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  try {
    const options: RequestInit & { duplex?: "half" } = {
      method: request.method, headers, signal: request.signal, cache: "no-store", redirect: "manual",
    };
    if (!["GET", "HEAD"].includes(request.method) && request.body) {
      options.body = request.body;
      options.duplex = "half";
    }
    const upstream = await fetch(target, options);
    const responseHeaders = new Headers();
    for (const name of ["content-type", "content-disposition", "content-range", "accept-ranges"]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    // Fetch may decompress: do not forward Content-Length/Encoding or hop-by-hop headers.
    responseHeaders.set("Cache-Control", "no-store");
    responseHeaders.set("X-Accel-Buffering", "no");
    // Preserve FastAPI trailing-slash redirects without exposing internal Docker DNS.
    const location = upstream.headers.get("location");
    if (location) {
      const redirect = new URL(location, target);
      if (redirect.origin === target.origin && redirect.pathname.startsWith("/api/")) {
        responseHeaders.set("Location", redirect.pathname + redirect.search);
      }
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    // Do not leak internal hostnames/errors; reconnecting SSE reads persisted Run state.
    return Response.json({ detail: "Backend unavailable" }, { status: 502 });
  }
}
