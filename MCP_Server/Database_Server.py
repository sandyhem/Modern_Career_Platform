# Database_Server.py
import contextlib
import os
from mcp.server.fastmcp import FastMCP, Context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import sqlparse
from pydantic import BaseModel

# ------------------ MCP APP ------------------
mcp = FastMCP("Database Server", stateless_http=True)

# ---------- DATABASE SETUP ----------
DATABASE_URL = os.getenv(
    "MYSQL_URL",
    "mysql+aiomysql://root:root@localhost:3306/recruit_db",
)
engine = create_async_engine(DATABASE_URL, echo=False)  # global engine

# Optional lifespan to verify DB connection and dispose engine
@contextlib.asynccontextmanager
async def lifespan(app: FastMCP):
    # Test connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))
    print("✅ DB Connected")
    yield
    await engine.dispose()
    print("🔒 DB Closed")

mcp.lifespan = lifespan

# ---------- RESOURCES ----------
@mcp.resource("file://schema/candidates")
def candidates_schema():
    """Database schema for candidates"""
    return {
        "table": "candidates",
        "columns": {
            "id": "INT",
            "name": "VARCHAR(100)",
            "cgpa": "FLOAT",
            "skills": "TEXT",
            "experience": "INT",
        },
    }

# ---------- PROMPT ----------
@mcp.prompt("sql_generator")
def sql_prompt():
    """Prompt template for generating SQL and executing it."""
    return """
    You are an SQL expert.
    Convert the following recruiter request into a valid MySQL SELECT query.
    Use only the provided table and columns.

    Table: candidates
    Columns: id, name, cgpa, skills, experience

    After generating SQL, execute it by calling the `execute_sql` tool with the query string.
    Return the results from the tool.
    """

# ---------- TOOL ----------
@mcp.tool("execute_sql")
async def execute_sql(ctx: Context, sql: str):
    """Safely executes SELECT queries."""
    # Parse and validate only SELECT statements
    parsed = sqlparse.parse(sql)
    for stmt in parsed:
        if stmt.get_type() != "SELECT":
            raise ValueError("❌ Only SELECT queries are allowed.")

    # Use global engine
    global engine

    async with engine.connect() as conn:
        result = await conn.execute(text(sql))
        rows = result.fetchall()
        cols = result.keys()

    return {"columns": cols, "rows": [dict(zip(cols, r)) for r in rows]}


@mcp.tool("create_user")
async def execute_sql(ctx: Context, sql: str):
    # Use global engine
    global engine

    async with engine.connect() as conn:
        result = await conn.execute(text(sql))
        rows = result.fetchall()
        cols = result.keys()

    return {"columns": cols, "rows": [dict(zip(cols, r)) for r in rows]}

# ---------- RUN MCP (optional standalone) ----------
# if __name__ == "__main__":
#     print("🚀 MCP Database Server running on http://localhost:10000")
#     mcp.serve(host="0.0.0.0", port=10000)
