# ─── LangGraph Planner-Executor — Entry Point ─────────────────────────────────
# Usage:
#   1. Start the weather server first:  python Tools/weather_server.py
#   2. Run this script:                 python main.py

import asyncio
import sys
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from graph import build_graph

# ─── LLM Configuration ────────────────────────────────────────────────────────
# Uncomment ONE of the following LLM options:

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# from langchain_ollama import ChatOllama
# llm = ChatOllama(model="llama3", temperature=0)


# ─── MCP Server Configuration ─────────────────────────────────────────────────

TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tools")

MCP_SERVERS = {
    "math": {
        "command": sys.executable,
        "args": [os.path.join(TOOLS_DIR, "math_server.py")],
        "transport": "stdio",
    },
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


# ─── Main ──────────────────────────────────────────────────────────────────────

async def main():
    """Set up MCP tools, build the LangGraph, and execute the test case."""

    # Connect to all MCP servers
    client = MultiServerMCPClient(MCP_SERVERS)

    # Load tools from each server
    tools = []
    for server_name in MCP_SERVERS:
        server_tools = await client.get_tools(server_name=server_name)
        tools.extend(server_tools)

    tools_map = {t.name: t for t in tools}
    print(f"MCP tools loaded: {list(tools_map.keys())}\n")

    # Build the LangGraph workflow
    graph = build_graph(llm, tools_map)

    # ── Test Case ──────────────────────────────────────────────────────────
    city = "New York"   # ← change this to any supported city
    goal = (
        f"Plan an outdoor event for 150 people in {city}: "
        "calculate how many tables are needed (8 people per table), "
        f"search the web for average outdoor event ticket prices, check current weather for {city}, "
        "and write a final summary."
    )

    # Execute the graph with initial state
    result = await graph.ainvoke({
        "goal": goal,
        "plan": [],
        "current_step": 0,
        "results": [],
    })

    # ── Display Final Output ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")
    for r in result["results"]:
        print(f"\n  Step {r['step']}: {r['description']}")
        print(f"    {r['result'][:]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
