#!/usr/bin/env python
"""
Test script to verify time-series-mcp server works with mcp-use framework.
"""
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


async def main():
    # Load environment variables from .env file (override=True ensures .env takes precedence)
    load_dotenv(override=True)
    print("📁 Loaded environment variables from .env file")

    # Build MCP server config dynamically (using current project directory)
    script_dir = Path(__file__).parent
    project_root = (script_dir / "../..").resolve()

    config = {
        "mcpServers": {
            "time-series-mcp": {
                "command": "uv",
                "args": [
                    "--directory",
                    str(project_root),
                    "run",
                    "time-series-mcp"
                ]
            }
        }
    }

    print(f"🔧 Creating MCP Client from config (project root: {project_root})...")
    # Create MCPClient from the config
    client = MCPClient.from_dict(config)

    # Create LLM (you'll need OPENAI_API_KEY in your .env file)
    print("\n🤖 Creating AI agent...")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment.")
        print("Please set it in .env file to test with AI agent.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    print(f"🔑 Using API key: {api_key[:5]}...{api_key[-5:]}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agent - this will connect to the MCP server
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    # Run a test queries
    queries = ["List all the available time series forecasting tools you have access to and briefly describe what each one does",
               "List all the available time series forecasting resources you have access to and briefly describe what each one does"]
    for query in queries: 
        print(f"\n💬 Sending query: {query}\n")

        result = await agent.run(query)
        print(f"\n📊 Result:\n{result}")

        print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
