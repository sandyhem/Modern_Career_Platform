import os
import contextlib
from mcp.server.fastmcp import FastMCP, Context
from pymongo import MongoClient
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ------------------ MCP APP ------------------
mcp = FastMCP("Database Server", stateless_http=True)

# ---------- DATABASE SETUP ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "recruitmentDB")
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

# ---------- LIFESPAN ----------
@contextlib.asynccontextmanager
async def lifespan(app: FastMCP):
    try:
        # Test connection
        client.list_database_names()
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
    yield
    client.close()
    print("🔒 MongoDB connection closed")

mcp.lifespan = lifespan

# ---------- RESOURCE ----------
@mcp.resource("file://schema/jobs")
def jobs_schema():
    """MongoDB jobs collection schema"""
    return {
        "collection": "jobs",
        "fields": {
            "title": "string",
            "type": "string",
            "department": "string",
            "location": "string",
            "postDate": "datetime",
            "endDate": "datetime",
            "responsibilities": "string",
            "qualifications": "string",
            "skills": "string"
        },
    }

# ---------- PROMPT ----------
@mcp.prompt("mongo_query_generator")
def mongo_prompt():
    """Prompt template for MongoDB queries"""
    return """
    You are an expert in MongoDB.
    Convert the recruiter’s request into a valid MongoDB find() query.
    Use only the provided collection and fields.
    After generating the query, call the `read_mongo` tool with the collection name and filter.
    """

# ---------- TOOL ----------
@mcp.tool("read_mongo")
def read_mongo(ctx: Context, collection_name: str, filter_query: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Reads data from a MongoDB collection.
    
    Args:
        collection_name (str): MongoDB collection name
        filter_query (dict, optional): MongoDB filter query
        limit (int): Max number of documents

    Returns:
        List[Dict]: List of documents
    """
    if filter_query is None:
        filter_query = {}

    collection = db[collection_name]
    results = list(collection.find(filter_query).limit(limit))

    # Convert ObjectId to string
    for doc in results:
        doc["_id"] = str(doc["_id"])

    return results

# ---------- TOOL 2: Insert document ----------
class MongoDoc(BaseModel):
    data: Dict[str, Any]
    collection_name: str

@mcp.tool("insert_mongo")
def insert_mongo(ctx: Context, doc: MongoDoc):
    """
    Insert a document into MongoDB
    """
    collection = db[doc.collection_name]
    result = collection.insert_one(doc.data)
    return {"_id": str(result.inserted_id), "message": "Document inserted successfully"}

# ---------- RUN MCP ----------
if __name__ == "__main__":
    print("🚀 MCP MongoDB Server running on http://localhost:10000")
    mcp.serve(host="0.0.0.0", port=10000)
