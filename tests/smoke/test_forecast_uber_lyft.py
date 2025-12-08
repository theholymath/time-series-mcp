#!/usr/bin/env python
"""
Test time series forecasting using the Uber & Lyft cab ride dataset.
This dataset contains ride price data from Boston in November 2018.
"""
import os
import asyncio
from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


def prepare_lyft_uber_dataset():
    """
    Load the uber Lyft  dataset 

   Uber and Lyft's ride prices are not constant like public transport. They are greatly affected by the demand and supply of rides at a given time. So what exactly drives this demand? The first guess would be the time of the day; times around 9 am and 5 pm should see the highest surges on account of people commuting to work/home. Another guess would be the weather; rain/snow should cause more people to take rides.
NOTE: The date is for simulated rides with real prices i.e. how much would the ride cost IF someone actually took it. Uber/Lyft DO NOT make this data public and nor is the case in this dataset

Content
With no public data of rides/prices shared by any entity, we tried to collect real-time data using Uber&Lyft api queries and corresponding weather conditions. We chose a few hot locations in Boston from this map
We built a custom application in Scala to query data at regular intervals and saved it to DynamoDB. The project can be found here on GitHub
We queried cab ride estimates every 5 mins and weather data every 1 hr.

The data is approx. for a week of Nov '18 ( I actually have included data collected while I was testing the 'querying' application so might have data spread out over more than a week. I didn't consider this as a time-series problem so did not worry about regular interval. The chosen interval was to query as much as data possible without unnecessary redundancy. So data can go from end week of Nov to few in Dec)

The Cab ride data covers various types of cabs for Uber & Lyft and their price for the given location. You can also find if there was any surge in the price during that time.
Weather data contains weather attributes like temperature, rain, cloud, etc for all the locations taken into consideration.

Inspiration
Our aim was to try to analyze the prices of these ride-sharing apps and try to figure out what factors are driving the demand. Do Mondays have more demand than Sunday at 9 am? Do people avoid cabs on a sunny day? Was there a Red Sox match at Fenway that caused more people coming in? We have provided a small dataset as well as a mechanism to collect more data. We would love to see more conclusions drawn.
    """
    print("📦 Checking Uber & Lyft dataset ...")

    # Define the dataset path (relative to this script)
    script_dir = Path(__file__).parent
    df_path = script_dir / "../../data/cab_weather/cab_rides.csv"

    # Check if the dataset exists
    if not df_path.exists():
        print(f"⚠️  Dataset not found at: {df_path}")
        print("Please download the Uber & Lyft dataset and place it at the above path.")
        print("Dataset source: https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices")
        return None

    # Load the dataset to verify it
    df = pd.read_csv(df_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n📊 First few rows:")
    print(df.head())

    return str(df_path.absolute())


async def test_cab_ride_demand_forecast():
    """
    Test the MCP time series forecasting tools with the Uber & Lyft dataset.
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
    dataset_path = prepare_lyft_uber_dataset()
    if dataset_path is None:
        print("❌ Test skipped: dataset not available")
        return

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

    This is the Uber & Lyft cab ride dataset from Boston in November 2018.
    It contains ride prices and weather data collected every 5 minutes.

    Please:
    1. Load the dataset and analyze it to determine the appropriate time and target columns
    2. Create a forecast for cab ride demand or prices
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

    print("\n✅ Uber & Lyft cab ride forecast test complete!")

    # Check if any output files were created
    output_dir = Path("output")
    if output_dir.exists():
        output_files = list(output_dir.glob("*"))
        if output_files:
            print(f"\n📁 Generated {len(output_files)} output file(s):")
            for f in output_files:
                print(f"   - {f}")


async def main():
    """Run the Uber & Lyft cab ride demand forecast test."""
    await test_cab_ride_demand_forecast()


if __name__ == "__main__":
    asyncio.run(main())
