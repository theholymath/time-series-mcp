#!/usr/bin/env python
"""
Test script to verify that resources are being registered correctly.
This tests the fix for MCP resource registration when loading datasets.
"""
import os
import json
import asyncio
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


def create_sample_data():
    """Create a simple sample CSV for testing."""
    print("📝 Creating sample time series data...")

    # Create simple sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'date': dates,
        'value': [100 + i * 2 + (i % 7) * 5 for i in range(30)]
    }
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = Path("data/sample_test_data.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"   ✅ Created sample data: {output_path}")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")

    return str(output_path.absolute())


async def test_resource_registration():
    """
    Test that:
    1. Resources list is empty initially
    2. After loading a dataset, it appears as a resource
    3. The resource can be read
    """
    # Load environment variables
    load_dotenv()

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment.")
        print("Please set it in .env file to test with AI agent.")
        return

    # Create sample data
    dataset_path = create_sample_data()

    # Load MCP server config
    config_path = "config/time_series_mcp_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n" + "="*70)
    print("🔧 TEST: Resource Registration")
    print("="*70)

    # Create MCPClient
    client = MCPClient.from_dict(config)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create agent
    agent = MCPAgent(llm=llm, client=client, max_steps=10)

    print("\n1️⃣  STEP 1: Check initial resources (should be empty or minimal)")
    print("-" * 70)

    result1 = await agent.run("List all available resources.")
    print(f"Initial resources:\n{result1}\n")

    print("\n2️⃣  STEP 2: Load a dataset")
    print("-" * 70)

    load_query = f"""
    Load the dataset from path: {dataset_path}
    Name it: 'test_sample'
    The date column is 'date' and the target column is 'value'.
    """

    result2 = await agent.run(load_query)
    print(f"Load result:\n{result2}\n")

    print("\n3️⃣  STEP 3: List resources again (should include ts://data/test_sample)")
    print("-" * 70)

    result3 = await agent.run("List all available resources. Tell me specifically if you see a resource with URI 'ts://data/test_sample'.")
    print(f"Resources after loading:\n{result3}\n")

    print("\n4️⃣  STEP 4: Try to read the dataset resource")
    print("-" * 70)

    result4 = await agent.run("Can you read the resource 'ts://data/test_sample' and show me the first few rows?")
    print(f"Read resource result:\n{result4}\n")

    print("\n" + "="*70)
    print("✅ Resource Registration Test Complete!")
    print("="*70)

    # Validate
    if "ts://data/test_sample" in result3:
        print("\n✅ SUCCESS: Dataset resource is registered correctly!")
    else:
        print("\n❌ FAILURE: Dataset resource was NOT found in resource list!")
        print("   This suggests the resource registration fix may not be working.")


async def main():
    """Run the resource registration test."""
    await test_resource_registration()


if __name__ == "__main__":
    asyncio.run(main())
