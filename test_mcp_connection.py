#!/usr/bin/env python
"""
Simple test to verify MCP server connection and tool listing.
No API key required.
"""
import asyncio
import json
from pathlib import Path
from mcp_use import MCPClient


async def main():
    # Load MCP server config
    config_path = Path(__file__).parent / "config/time_series_mcp_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("📋 Config loaded:")
    print(json.dumps(config, indent=2))

    print("\n🔧 Creating MCP Client from config...")
    client = MCPClient.from_dict(config)

    print("\n🔌 Creating sessions with MCP servers...")
    await client.create_all_sessions()

    print("\n📊 Getting server session...")
    session = client.get_session("time-series-mcp")

    if session is None:
        print("❌ Failed to get session for 'time-series-mcp'")
        print(f"Available sessions: {list(client._sessions.keys())}")
        return

    print("✅ Session obtained!")

    print("\n🔍 Listing available tools...")
    tools = await session.list_tools()

    print(f"\n📦 Found {len(tools)} tools:")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        if hasattr(tool, 'description'):
            # Print first line of description
            desc_line = tool.description.split('\n')[0].strip()
            print(f"   {desc_line[:80]}")

    print("\n🔍 Listing available resources...")
    resources = await session.list_resources()

    print(f"\n📦 Found {len(resources)} resources:")
    for i, resource in enumerate(resources, 1):
        print(f"{i}. {resource.name}")

    print("\n✅ MCP Server is connected and working!")

    await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(main())
