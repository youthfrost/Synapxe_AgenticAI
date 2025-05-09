from datetime import datetime
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from openai_sample_network import agents_network

mcp = FastMCP("Agent Networks")

# Expose the functions as MCP tools
@mcp.tool()
async def healthcare_finance_domain(query: str) -> str:
    """
    Process requests relating to healthcare financing domain, including questions relating to healthcare financing and claims records.
    
    Args:
        query: The user query or request
        
    Returns:
        Response as a string
    """
    try:
        result = await agents_network(query)
        print(f"*** Result of MCP tool that invokes the agent network: {result}")
        return str(result)
    except Exception as e:
        print(f"Error in healthcare financing domain: {e}")
        return f"Error occurred when calling agents network: {e}"
    
if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run(transport="sse")