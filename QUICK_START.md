# Quick Start Guide for time-series-mcp

This guide helps you get started with the Time Series MCP server in under 5 minutes.

## Prerequisites

- Python 3.11+
- `uv` package manager (will help you install if needed)

## Installation (3 steps)

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install uv
```

### 2. Clone and Setup

```bash
git clone https://github.com/yourusername/time-series-mcp.git
cd time-series-mcp
uv sync
```

### 3. Configure Claude Desktop

**Find your project path:**
```bash
pwd
```

**Copy and edit the config:**
```bash
cp .mcp.json.sample .mcp.json
# Edit .mcp.json - replace /ABSOLUTE/PATH/TO/time-series-mcp with output from pwd
```

**Add to Claude Desktop config:**

Open `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and paste:

```json
{
  "mcpServers": {
    "time-series-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/YOUR/ACTUAL/PATH/time-series-mcp",
        "run",
        "time-series-mcp"
      ],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Restart Claude Desktop** and you're done! 🎉

## Test It

Ask Claude:
> "List the available time series datasets"

If you see tools available (hammer icon), it's working!

## Example Usage

> "Load my sales data from sales_2024.csv and create a 2-month forecast"

## Troubleshooting

### "MCP server not showing up"
- Verify your path is absolute (no `~` or `.`)
- Check Claude Desktop logs: `~/Library/Logs/Claude/` (macOS)

### "Command not found: uv"
- Run `which uv` to verify installation
- Try closing and reopening your terminal

### "Python version error"
- Run `python --version` - must be 3.11+
- Update Python if needed

## Next Steps

See [README.md](README.md) for:
- Alternative configuration methods (pip, venv)
- Full list of available tools
- Advanced usage examples
