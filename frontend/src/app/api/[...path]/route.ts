import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 180;

const BACKEND_URL = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const PROXY_TIMEOUT_MS = 180_000;

async function proxyRequest(
  request: NextRequest,
  context: { params: { path: string[] } }
): Promise<NextResponse> {
  const pathSegments = context.params.path || [];
  const targetUrl = new URL(`${BACKEND_URL}/${pathSegments.join("/")}`);
  targetUrl.search = request.nextUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) {
    headers.set("content-type", contentType);
  }
  const accept = request.headers.get("accept");
  if (accept) {
    headers.set("accept", accept);
  }

  const hasBody = !["GET", "HEAD"].includes(request.method.toUpperCase());
  const body = hasBody ? await request.text() : undefined;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROXY_TIMEOUT_MS);

  try {
    const upstream = await fetch(targetUrl.toString(), {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      signal: controller.signal,
    });

    const respText = await upstream.text();
    const respHeaders = new Headers();
    const upstreamContentType = upstream.headers.get("content-type");
    if (upstreamContentType) {
      respHeaders.set("content-type", upstreamContentType);
    }

    return new NextResponse(respText, {
      status: upstream.status,
      headers: respHeaders,
    });
  } catch (err) {
    const isAbort = err instanceof Error && err.name === "AbortError";
    return NextResponse.json(
      {
        detail: isAbort
          ? `后端处理超时（>${PROXY_TIMEOUT_MS / 1000}s），请稍后重试`
          : `无法连接后端服务 (${BACKEND_URL}): ${err instanceof Error ? err.message : String(err)}`,
      },
      { status: isAbort ? 504 : 502 }
    );
  } finally {
    clearTimeout(timer);
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PUT = proxyRequest;
export const PATCH = proxyRequest;
export const DELETE = proxyRequest;
