# ─── PLANNER-EXECUTOR AGENT with MCP Tools ────────────────────────────────────

import asyncio
import json
import re
import sys
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
load_dotenv()

# ─── LLM ─────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ─── MCP Server Configuration ─────────────────────────────────────────────────
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tools")

MCP_SERVERS = {
    "search": {
        "command": sys.executable,
        "args": [os.path.join(TOOLS_DIR, "search_server.py")],
        "transport": "stdio",
    },
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    },
}

# ─── Tool helper ──────────────────────────────────────────────────────────────
async def get_mcp_tools():
    """Connect to MCP servers and return (tools list, tools_map dict)."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = []
    for server_name in MCP_SERVERS:
        server_tools = await client.get_tools(server_name=server_name)
        tools.extend(server_tools)
    tools_map = {t.name: t for t in tools}
    return tools, tools_map

# ─── Planner prompt ───────────────────────────────────────────────────────────
PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available MCP tools and their EXACT argument names:
  - search_web(query: str)                     → search the web or Wikipedia for any topic
  - search_news(query: str)                    → search for latest news
  - get_current_weather(city: str)             → get current weather for a city
  - get_weather_forecast(city: str, days: int) → get weather forecast for N days

Rules:
  - Use AT MOST ONE search tool call per topic. Do not search the same topic multiple times.
  - Keep the total number of steps as small as possible (ideally 3-5).
  - Use null for tool/args on synthesis or writing steps.

Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "search_web":           "query",
    "search_news":          "query",
    "get_current_weather":  "city",
    "get_weather_forecast": "city",
}

# ─── Result normaliser ───────────────────────────────────────────────────────
def extract_text(result) -> str:
    """Unwrap MCP content-block list to a plain string."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in result
        )
    return str(result)

# ─── Arg safety ───────────────────────────────────────────────────────────────
def safe_args(tool_name: str, raw_args: dict) -> dict:
    """Remap hallucinated arg names to the correct parameter."""
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    print(f"  [!] Remapped {raw_args} → {{'{expected}': '{first_val}'}}")
    return {expected: str(first_val)}

# ─── Planner-Executor ─────────────────────────────────────────────────────────
async def planner_executor_mcp(goal: str) -> list:
    print(f" Goal: {goal}\n")
    _, tools_map = await get_mcp_tools()
    print(f" MCP tools loaded: {list(tools_map.keys())}\n")

    # ── Phase 1: Plan ──────────────────────────────────────────────────────────
    plan_resp = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=goal),
    ])
    raw_text  = plan_resp.content if isinstance(plan_resp.content, str) else plan_resp.content[0].get("text", "")
    clean_json = re.sub(r"```json|```", "", raw_text).strip()
    plan = json.loads(clean_json)

    print(f" Plan ({len(plan)} steps):")
    for s in plan:
        print(f"  Step {s['step']}: {s['description']} | tool={s.get('tool')}")
    print()

    # ── Phase 2: Execute ───────────────────────────────────────────────────────
    results = []
    for step in plan:
        print(f"  Step {step['step']}: {step['description']}")
        tool_name = step.get("tool")

        if tool_name and tool_name in tools_map:
            corrected = safe_args(tool_name, step.get("args") or {})
            result = extract_text(await tools_map[tool_name].ainvoke(corrected))
        else:
            # Synthesis step — LLM must use ONLY the tool results; no re-derivation
            context  = "\n".join(f"Step {r['step']}: {r['result']}" for r in results)
            response = llm.invoke([
                SystemMessage(content=(
                    "You are a synthesis agent. "
                    "Use ONLY the exact data from the Context below. "
                    "Do NOT invent, recalculate, or add information not present in the Context."
                )),
                HumanMessage(content=f"{step['description']}\n\nContext:\n{context}"),
            ])
            result = response.content

        print(f"    {result[:]}\n")
        results.append({"step": step["step"], "description": step["description"], "result": result})

    return results

# ─── Entry point ──────────────────────────────────────────────────────────────
async def run():
    results = await planner_executor_mcp(
        "Search for Q3 2024 tech-industry sales trends and look up the history of LangChain, "
        "then summarize both."
    )

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"\n  Step {r['step']}: {r['description']}")
        print(f"  {'-'*56}")
        print(f"  {r['result']}")
    print()

if __name__ == "__main__":
    asyncio.run(run())