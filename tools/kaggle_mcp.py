#!/usr/bin/env python3
"""Lightweight CLI for calling tools on the Kaggle MCP server (https://www.kaggle.com/mcp).

Usage:
  python3 kaggle_mcp.py <tool_name> ['{"request": {...}}']
  python3 kaggle_mcp.py tools            # list all available tools
  python3 kaggle_mcp.py --raw <tool> ... # print raw JSON instead of extracted text

Examples:
  python3 kaggle_mcp.py get_competition '{"request": {"competitionName": "titanic"}}'
  python3 kaggle_mcp.py list_competition_data_files '{"request": {"competitionName": "titanic"}}'
  python3 kaggle_mcp.py get_competition_leaderboard '{"request": {"competitionName": "titanic", "pageSize": 10}}'

Auth (only needed for non-public operations):
  export KAGGLE_API_TOKEN=...   # or put the token in ~/.kaggle/access_token
"""
import json
import os
import sys
import urllib.request

MCP_URL = "https://www.kaggle.com/mcp"


def get_token():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()
    path = os.path.expanduser("~/.kaggle/access_token")
    if os.path.exists(path):
        return open(path).read().strip()
    return None


def rpc(method, params=None, msg_id=1):
    body = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        body["params"] = params
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"},
    )
    token = get_token()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        raw = resp.read().decode()
    # The response is SSE: one JSON object per `data:` line (possibly chunked).
    data = "".join(l[5:] for l in raw.splitlines() if l.startswith("data:"))
    dec = json.JSONDecoder()
    idx, objs = 0, []
    while idx < len(data):
        while idx < len(data) and data[idx] in " \n":
            idx += 1
        if idx >= len(data):
            break
        obj, idx = dec.raw_decode(data, idx)
        objs.append(obj)
    return objs[-1]


def list_tools():
    resp = rpc("tools/list")
    for t in resp["result"]["tools"]:
        desc = (t.get("description") or "").replace("\n", " ")[:80]
        print(f"{t['name']:45s} {desc}")


def call_tool(name, arguments, raw=False):
    resp = rpc("tools/call", {"name": name, "arguments": arguments})
    result = resp.get("result", {})
    contents = result.get("content", [])
    texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
    if result.get("isError"):
        print(f"ERROR from {name}: {' | '.join(texts)}", file=sys.stderr)
        sys.exit(1)
    if raw:
        print(json.dumps(result, indent=2))
        return
    for text in texts:
        # pretty-print if the text is itself JSON
        try:
            print(json.dumps(json.loads(text), indent=2))
        except (json.JSONDecodeError, TypeError):
            print(text)


def main():
    args = sys.argv[1:]
    raw = "--raw" in args
    args = [a for a in args if a != "--raw"]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "tools":
        list_tools()
        return
    tool = args[0]
    arguments = json.loads(args[1]) if len(args) > 1 else {}
    call_tool(tool, arguments, raw=raw)


if __name__ == "__main__":
    main()
