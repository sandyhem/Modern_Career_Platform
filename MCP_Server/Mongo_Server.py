import contextlib
import os
from mcp.server.fastmcp import FastMCP, Context
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Any

# ------------------ MCP APP ------------------
mcp = FastMCP("MongoDB Resume Server", stateless_http=True)

# ---------- MONGODB CONNECTION ----------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/resume_database"  # Change this to your actual URI
)

client = MongoClient(MONGO_URI)
db = client["edu_tech"]
resumes = db["resumes"]

# ---------- LIFESPAN ----------
@contextlib.asynccontextmanager
async def lifespan(app: FastMCP):
    try:
        client.admin.command("ping")
        print("✅ MongoDB Connected Successfully")
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        raise
    yield
    client.close()
    print("🔒 MongoDB Connection Closed")

mcp.lifespan = lifespan

# ---------- RESOURCE SCHEMA ----------
@mcp.resource("file://schema/resumes")
def resumes_schema():
    """Schema definition for resumes collection"""
    return {
        "collection": "resumes",
        "fields": {
            "name": "string",
            "contact_information.email": "string",
            "contact_information.phone": "string",
            "skills": "list[string]",
            "education": "list[string]",
            "experience": "integer",
            "resume_url": "string",
        },
        "description": "Parsed student resume data stored in MongoDB (read-only)."
    }

# ---------- PROMPT ----------
@mcp.prompt("mongodb_query_prompt")
def mongodb_query_prompt():
    """
    Prompt template for generating MongoDB queries.
    """
    return """
    You are a MongoDB query assistant.
    Convert the recruiter request into a valid MongoDB filter (as JSON).
    Only use the collection `resumes` with the following fields:

    - name (string)
    - contact_information.email (string)
    - contact_information.phone (string)
    - skills (list[string])
    - education (list[string])
    - experience (integer)
    - resume_url (string)

    Example queries:
    - Find resumes with skill 'Python' => {"skills": "Python"}
    - Find resumes with experience greater than 3 => {"experience": {"$gt": 3}}
    - Find resumes with both Python and React skills => {"skills": {"$all": ["Python", "React"]}}

    After generating the filter, call the `execute_mongo_query` tool
    with that JSON object as the `query` argument.
    """

# ---------- TOOL ----------
def is_read_only_query(query: Any) -> bool:
    """Check that the query contains no write operators."""
    forbidden_ops = {"$set", "$unset", "$update", "$delete", "$insert", "$push", "$pull"}
    return not any(op in str(query).lower() for op in forbidden_ops)

@mcp.tool("execute_mongo_query")
async def execute_mongo_query(ctx: Context, query: dict):
    """
    Executes a safe MongoDB find() operation using a given query filter.
    """
    if not is_read_only_query(query):
        return {"error": "❌ Write operations are not allowed. Only find queries are permitted."}

    try:
        print(f"Executing MongoDB query: {query}")
        results = list(resumes.find(query, {"_id": 0}))
        return {
            "count": len(results),
            "results": results
        }
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}

# ---------- RUN MCP SERVER (optional standalone) ----------
# if __name__ == "__main__":
#     print("🚀 MongoDB MCP Server running on http://localhost:10000")
#     mcp.serve(host="0.0.0.0", port=10000)
