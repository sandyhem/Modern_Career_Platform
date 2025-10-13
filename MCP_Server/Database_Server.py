# mongo_query_tool.py
from mcp.server.fastmcp import FastMCP, Context
from pymongo import MongoClient
from typing import Any, Dict, List
import json

mcp = FastMCP("MongoDB Query Executor", stateless_http=True)

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017")
recruitment_db = client["recruitmentDB"]
job_applicants_collection = recruitment_db["jobapplicants"]

import json


# ---------- RESOURCES ----------
@mcp.resource("schema://jobapplicants")
def get_schema() -> str:
    """
    Returns the schema of the jobapplicants collection based on the applicant_serializer.
    This helps the LLM understand the data structure for query generation.
    """
    schema = {
        "collection": "jobapplicants",
        "description": "Collection storing job applicant information with GitHub, LeetCode, Codeforces scores, badges, and metadata",
        "fields": {
            # Basic identifiers
            "id": {"type": "string", "description": "Unique identifier of the applicant document"},
            "jobId": {"type": "string", "description": "ID of the job applied for"},
            "studentId": {"type": "string", "description": "ID of the student/applicant"},

            # GitHub fields
            "github_username": {"type": "string", "description": "GitHub username"},
            "github_job_description_preview": {"type": "string", "description": "Job description snippet matched from GitHub"},
            "github_languages": {"type": "object", "description": "Dictionary of programming languages and contribution counts", "keys": "string", "values": "number"},
            "github_matched_keywords": {"type": "array", "description": "List of keywords matched in GitHub repositories", "items": "string"},
            "github_commit_factor": {"type": "number", "description": "Normalized commit activity factor"},
            "github_match_score": {"type": "number", "description": "Score indicating GitHub match quality (0-1)"},
            "github_compatibility_score": {"type": "number", "description": "Compatibility score with job requirements (0-100)"},
            "total_repos": {"type": "number", "description": "Total number of repositories"},
            "total_stars": {"type": "number", "description": "Total GitHub stars"},
            "total_forks": {"type": "number", "description": "Total forks"},
            "total_commits": {"type": "number", "description": "Total commits"},

            # LeetCode fields
            "leetcode_username": {"type": "string", "description": "LeetCode username"},
            "leetcode_country": {"type": "string", "description": "Country of the LeetCode user"},
            "leetcode_ranking": {"type": "number", "description": "Global LeetCode ranking"},
            "leetcode_totalproblems": {"type": "number", "description": "Total problems solved"},
            "leetcode_easy": {"type": "number", "description": "Number of easy problems solved"},
            "leetcode_medium": {"type": "number", "description": "Number of medium problems solved"},
            "leetcode_hard": {"type": "number", "description": "Number of hard problems solved"},
            "leetcode_acceptanceRate": {"type": "string", "description": "Acceptance rate as a percentage string"},
            "leetcode_contests": {"type": "number", "description": "Number of contests participated"},
            "badges": {"type": "array", "description": "List of badges earned", "items": {
                "id": "string",
                "name": "string",
                "icon": "string",
                "earnedOn": "string"
            }},

            # Codeforces fields
            "codeforces_handle": {"type": "string", "description": "Codeforces username"},
            "codeforces_rating": {"type": "number", "description": "Current Codeforces rating"},
            "codeforces_rank": {"type": "string", "description": "Current Codeforces rank"},
            "codeforces_maxRating": {"type": "number", "description": "Maximum rating achieved on Codeforces"},
            "codeforces_maxRank": {"type": "string", "description": "Maximum rank achieved on Codeforces"}
        },
        "indexes": ["studentId", "jobId", "github_username", "leetcode_username"],
        "mongodb_operators": {
            "comparison": ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"],
            "logical": ["$and", "$or", "$not", "$nor"],
            "element": ["$exists", "$type"],
            "array": ["$all", "$elemMatch", "$size"],
            "text": ["$regex", "$text"]
        }
    }

    return json.dumps(schema, indent=2)

@mcp.prompt()
def translate_query_prompt():
    """
    Prompt template for translating natural language to MongoDB queries.
    """
    return [
        {
"""You are a MongoDB query translator. Your task is to convert natural language questions into valid MongoDB query filters (Python dict format).
**SCHEMA REFERENCE:**
Use the schema from resource to understand available fields and their types.

**TRANSLATION RULES:**
1. Output ONLY a valid Python dictionary that can be used with pymongo's find() method
2. Use MongoDB query operators: $gt, $lt, $gte, $lte, $eq, $ne, $in, $nin, $and, $or, $regex, $exists
3. For text searches, use case-insensitive regex: {"field": {"$regex": "pattern", "$options": "i"}}
4. For array fields (like skills), use $in or $all operators
5. Combine multiple conditions using $and or $or
6. For date comparisons, use ISO format strings or ISODate syntax
7. Return empty dict {} for queries requesting all documents

**EXAMPLES:**

Query: "Find applicants with github score above 0.8"
Output: {"github_match_score": {"$gt": 0.8}}

Query: "Show me Python developers in pending status"
Output: {"$and": [{"skills": {"$regex": "python", "$options": "i"}}, {"status": "pending"}]}

Query: "Applicants with more than 5 years experience and overall score >= 0.7"
Output: {"$and": [{"experience_years": {"$gt": 5}}, {"overall_score": {"$gte": 0.7}}]}

Query: "Find candidates who applied for Senior Developer or Tech Lead positions"
Output: {"applied_position": {"$in": ["Senior Developer", "Tech Lead"]}}

Query: "Show all applicants"
Output: {}

**YOUR TASK:**
Translate the following natural language query into a MongoDB filter dictionary:
After generating filter, immediately call the tool `execute_query` with description "Execute a MongoDB query and return results" with the output and
give the output from the tool as your final answer.

Return ONLY the Python dictionary, no explanations or markdown formatting."""
        }
    ]

@mcp.tool(description="Execute a MongoDB query and return results")
def execute_query(query: Dict[str, Any], ctx: Context):
    """
    Execute a MongoDB query directly from a JSON object.
    Example Input:
      {
        "github_match_score": {"$gt": 0.8}
      }
    """
    try:
        results = list(job_applicants_collection.find(query, {"_id": 0}))  # No _id in output
        return {"count": len(results), "data": results}
    except Exception as e:
        return {"error": str(e)}




