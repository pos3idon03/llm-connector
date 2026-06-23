import asyncio
import json
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


class _Conn:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.connected = False
        self.tools: list = []
        self.session: Optional[ClientSession] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.cfg['name']}")

    async def _run(self) -> None:
        try:
            if self.cfg["transport"] == "stdio":
                cm = stdio_client(StdioServerParameters(
                    command=self.cfg["command"],
                    args=self.cfg.get("args", []),
                ))
            else:
                cm = streamablehttp_client(self.cfg["url"])

            async with cm as transport:
                r, w = transport[0], transport[1]
                async with ClientSession(r, w) as sess:
                    await sess.initialize()
                    self.tools = (await sess.list_tools()).tools
                    self.session = sess
                    self.connected = True
                    await asyncio.sleep(float("inf"))  # ponytail: keep session alive until cancelled
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import logging
            causes = e.exceptions if isinstance(e, BaseExceptionGroup) else [e]
            for cause in causes:
                logging.getLogger(__name__).warning("MCP %s connection failed: %s", self.cfg["name"], cause)
        finally:
            self.connected = False
            self.session = None

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


class MCPManager:
    def __init__(self) -> None:
        self._conns: dict[str, _Conn] = {}
        self._path = Path("mcp_servers.json")

    async def load(self, path: str = "mcp_servers.json") -> None:
        self._path = Path(path)
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text())
        for cfg in data.get("servers", []):
            conn = _Conn(cfg)
            self._conns[cfg["name"]] = conn
            conn.start()

    async def shutdown(self) -> None:
        await asyncio.gather(*[c.stop() for c in self._conns.values()], return_exceptions=True)

    def get_all_tools(self) -> list[dict]:
        out = []
        for name, conn in self._conns.items():
            if not conn.connected:
                continue
            for t in conn.tools:
                out.append({
                    "type": "function",
                    "function": {
                        "name": f"{name}__{t.name}",
                        "description": t.description or "",
                        "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    },
                })
        return out

    async def call_tool(self, qualified: str, args: dict) -> str:
        server, tool = qualified.split("__", 1)
        conn = self._conns.get(server)
        if not conn or not conn.session:
            raise ValueError(f"Server {server!r} not connected")
        result = await conn.session.call_tool(tool, args)
        return "\n".join(
            item.text if hasattr(item, "text") else str(item)
            for item in result.content
        )

    def servers_status(self) -> list[dict]:
        out = []
        for name, conn in self._conns.items():
            cfg = conn.cfg
            out.append({
                "name": name,
                "transport": cfg["transport"],
                "endpoint": (
                    " ".join([cfg.get("command", "")] + cfg.get("args", []))
                    if cfg["transport"] == "stdio"
                    else cfg.get("url", "")
                ),
                "connected": conn.connected,
                "tool_count": len(conn.tools),
                "tools": [
                    {"name": t.name, "description": t.description or ""}
                    for t in conn.tools
                ],
            })
        return out

    def _save(self) -> None:
        self._path.write_text(json.dumps({"servers": [c.cfg for c in self._conns.values()]}, indent=2))

    async def add_server(self, cfg: dict) -> None:
        name = cfg["name"]
        if name in self._conns:
            await self.remove_server(name)
        conn = _Conn(cfg)
        self._conns[name] = conn
        self._save()
        conn.start()

    async def remove_server(self, name: str) -> None:
        conn = self._conns.pop(name, None)
        if conn:
            await conn.stop()
        self._save()
