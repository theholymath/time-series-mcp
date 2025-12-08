# Time-Series MCP

## Adding Server to an application

### Anthropic


#### Claude Code
```bash
claude mcp add time-series-mcp --scope user -- uv run /ABSOLUTE/PATH/TO/time-series-mcp-v2/server.py
```

#### Claude Desktop

Locate your config file:

* Mac: ~/Library/Application Support/Claude/claude_desktop_config.json
* Windows: %APPDATA%\Claude\claude_desktop_config.json

```json
{
  "mcpServers": {
    "time-series-mcp": {
      "command": "uv",
      "args": [
        "run",
        "server.py"
      ],
      "cwd": "/ABSOLUTE/PATH/TO/YOUR/time-series-mcp-v2",
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```


Restart Inspector: npx @modelcontextprotocol/inspector uv run server.py
