#!/usr/bin/env python
"""
Test time series forecasting using the statsmodels Longley dataset.
This dataset contains US macroeconomic data from 1947-1962.
"""
import os
import json
import asyncio
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


def prepare_longley_dataset():
    """
    Load the Longley dataset from statsmodels and prepare it for time series analysis.

    The Longley dataset contains:
    - TOTEMP: Total employment (target variable)
    - GNPDEFL: GNP deflator
    - GNP: Gross National Product
    - UNEMP: Number of unemployed
    - ARMED: Size of armed forces
    - POP: Population
    - YEAR: Year (1947-1962)
    """
    print("📦 Loading Longley dataset from statsmodels...")

    # Load the dataset
    data = sm.datasets.longley.load_pandas()
    df = data.data

    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n📊 First few rows:")
    print(df.head())

    # The Longley dataset doesn't have a date column, so we'll create one
    # Data is from 1947-1962 (16 years)
    df['YEAR'] = range(1947, 1947 + len(df))

    # Save to CSV for the MCP server (path relative to this script)
    script_dir = Path(__file__).parent
    output_path = script_dir / "../../data/longley_employment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n💾 Saved dataset to: {output_path}")
    print(f"   Target variable: TOTEMP (Total Employment)")
    print(f"   Time period: 1947-1962")

    return str(output_path.absolute())


async def test_longley_forecast():
    """
    Test the MCP time series forecasting tools with the Longley dataset.
    """
    # Load environment variables from .env file (override=True ensures .env takes precedence)
    load_dotenv(override=True)
    print("📁 Loaded environment variables from .env file")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in environment.")
        print("Please set it in .env file to test with AI agent.")
        return

    # Prepare the dataset
    dataset_path = prepare_longley_dataset()

    # Load MCP server config (path relative to this script)
    script_dir = Path(__file__).parent
    config_path = script_dir / "../../config/time_series_mcp_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("\n" + "="*70)
    print("🔧 Creating MCP Client and AI Agent...")
    print("="*70)

    # Create MCPClient from the config
    client = MCPClient.from_dict(config)

    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"🔑 Using API key: {api_key[:5]}...{api_key[-5:]}")

    # Create LLM - using gpt-4o for better performance
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Create agent - this will connect to the MCP server
    agent = MCPAgent(llm=llm, client=client, max_steps=15)

    # Construct the query - simplified to avoid context overflow
    query = f"""
    I have a time series dataset at this path: {dataset_path}

    This is the Longley dataset containing US macroeconomic data from 1947-1962.

    Please:
    1. Load the dataset (the date column is YEAR, and the target variable is TOTEMP - Total Employment)
    2. Create a forecast for the next 5 years (1963-1967) using the annual frequency
    3. Show me the forecast values with confidence intervals

    Use the quick_forecast tool or create_forecast to keep it simple.
    """

    print("\n" + "="*70)
    print("💬 QUERY:")
    print("="*70)
    print(query)

    print("\n" + "="*70)
    print("🤖 AI Agent Processing...")
    print("="*70 + "\n")

    # Run the query
    result = await agent.run(query)

    print("\n" + "="*70)
    print("📊 RESULT:")
    print("="*70)
    print(result)
    print("="*70)

    print("\n✅ Longley dataset forecast test complete!")

    # Check if any output files were created
    output_dir = Path("output")
    if output_dir.exists():
        output_files = list(output_dir.glob("*"))
        if output_files:
            print(f"\n📁 Generated {len(output_files)} output file(s):")
            for f in output_files:
                print(f"   - {f}")


async def main():
    """Run the Longley dataset forecast test."""
    await test_longley_forecast()


if __name__ == "__main__":
    asyncio.run(main())
