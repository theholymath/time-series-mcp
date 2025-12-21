from mcp.server.fastmcp import FastMCP

mcp = FastMCP("HelloWorldMCP")

@mcp.tool()
def my_awesome_tool(user_string:str) -> str:
    """
    Take a string and write it backwards to the user

    Args:
        user_string: the input string from the user 
    """

    return user_string[::-1]

    
if __name__ == "__main__":
    mcp.run()
