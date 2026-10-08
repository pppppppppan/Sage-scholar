"""Exercise one actual MCP call using the application's configuration, without an LLM."""
import argparse
import json
import socket
from pathlib import Path
from urllib.parse import urlsplit

from sage_scholar.config import Config
from .loader import iter_mcp_clients


def main():
    parser = argparse.ArgumentParser(description="单独检查 Fetch/Tavily 的真实 MCP 调用，不启动 Research 或模型。")
    parser.add_argument("--server", choices=("fetch", "tavily"), required=True)
    parser.add_argument("--url", default="https://arxiv.org/abs/1706.03762", help="Fetch 测试地址")
    parser.add_argument("--query", default="On-Policy Distillation OPD", help="Tavily 测试检索词；会消耗一次搜索 API 调用")
    args = parser.parse_args()
    config = Config()
    path = Path(config.config_dir) / "mcp_servers.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))[args.server]
        # Do not dump env: it may contain API keys. This only displays proxy host/port.
        proxy = settings.get("env", {}).get("https_proxy", "")
        if proxy and not proxy.startswith("${"):
            address = urlsplit(proxy)
            port = address.port or (443 if address.scheme == "https" else 80)
            with socket.create_connection((address.hostname, port), timeout=3):
                pass
            print(f"[OK] 代理端口可连接：{address.hostname}:{port}（外网连通性由下面的 MCP 调用验证）", flush=True)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] 配置读取或代理端口检查失败：{type(exc).__name__}。请检查代理软件及 {path}")
        return 1

    client = None
    try:
        for status, connected in iter_mcp_clients(path, server_names={args.server}, connect_timeout=config.mcp_connect_timeout):
            client = connected
            if client is None:
                print(f"[FAIL] {args.server} 初始化失败：{status.get('reason')}。缺少配置时请检查 .env；连接失败时查看上方 MCP 日志。")
                return 1
        if client is None:
            print("[FAIL] 没有加载到 MCP 服务")
            return 1
        print(f"[OK] {args.server} MCP 初始化成功，开始真实工具调用…", flush=True)
        if args.server == "fetch":
            name, parameters = "fetch", {"url": args.url, "max_length": 1000}
        else:
            name, parameters = "tavily_search", {"query": args.query, "search_depth": "basic", "max_results": 5}
        if name not in {tool.name for tool in client.tools}:
            raise ValueError(f"MCP 未提供 {name} 工具")
        result = client.call_tool(name, parameters, _max_retries=0)
        if not result.strip() or result.lstrip().lower().startswith(("error:", "<error>")):
            raise ValueError("工具返回为空或包含错误标记")
        print(f"[PASS] {name} 调用成功，返回 {len(result)} 个字符。")
        print(result[:600])
        return 0
    except Exception as exc:
        # A tool error is useful here, but never expose credentials embedded by a remote client.
        import os
        message = str(exc)
        key = os.getenv("TAVILY_API_KEY")
        if key:
            message = message.replace(key, "[REDACTED]")
        print(f"[FAIL] {args.server}: {message[:1000]}")
        return 1
    finally:
        if client is not None:
            client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
