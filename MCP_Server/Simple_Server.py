from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Simple MCP Server", stateless_http=True)


@mcp.tool(description="Simple tool to add two numbers.")
def add_numbers(n: int, m: int) -> int:
    return n * m