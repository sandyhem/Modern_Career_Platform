# app.py
import json
import asyncio
import os
import httpx
from datetime import datetime

from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
from typing import Dict, List, Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from fastapi import HTTPException
from bson import ObjectId
from pymongo import MongoClient

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="MCP FastAPI Client",
    description="A FastAPI client for Model Context Protocol with React integration",
    version="1.0.0"
)

# Configure CORS middleware to allow requests from React
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default dev server
        "http://localhost:3001",  # Alternative React port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://localhost:3000",
        "https://localhost:3001",
        "*"  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class CalcRequest(BaseModel):
    n: int
    m: int

class QueryRequest(BaseModel):
    query: str


# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for React frontend"""
    return {"status": "healthy", "timestamp": "2025-10-07"}

# Handle preflight requests for CORS
@app.options("/{full_path:path}")
async def preflight_handler(full_path: str):
    """Handle CORS preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )



@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    async with streamablehttp_client("http://localhost:10000/db/mcp") as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 🔹 Load MCP tools and convert to LangChain-compatible tools
            tools = await load_mcp_tools(session)

            print(f"🔧 Loaded {len(tools)} MCP tools")

            if not tools:
                return {"error": "No tools loaded from MCP server"}

            # Return tool information
            tool_info = []
            for tool in tools:
                tool_info.append({
                    "name": tool.name,
                    "description": tool.description,
                })
            
            return {"tools": tool_info, "count": len(tools)}


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query using MCP tools and AI agent"""
    if not request.query or not request.query.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "Query cannot be empty",
                "status": "error",
                "timestamp": "2025-10-07"
            }
        )
    
    async with streamablehttp_client("http://localhost:10000/db/mcp") as (read_stream, write_stream, get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 🔹 Load MCP tools and convert to LangChain-compatible tools
            tools = await load_mcp_tools(session)

            print(f"🔧 Loaded {len(tools)} MCP tools")

            if not tools:
                return {"error": "No tools loaded from MCP server"}

            # 🔹 Setup Gemini LLM
            google_api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                max_retries=2,
                google_api_key=google_api_key
            )

            # 🔹 Create LangGraph Agent
            agent = create_react_agent(llm, tools)

            # Use the user's query
            query = [
                {"role": "user", "content": request.query}
            ]

            try:
                # 🔹 Invoke the agent
                print(f"🔍 Processing query: {request.query}")
                response = await agent.ainvoke({"messages": query})
                # print(response)

                # ✅ Extract only final textual response
                final_output = response["messages"][-1].content

                print("🤖 Final Output:", final_output)
                return JSONResponse(
                    status_code=200,
                    content={
                        "result": final_output, 
                        "query": request.query,
                        "status": "success",
                        "timestamp": "2025-10-07"
                    }
                )
            
            except Exception as e:
                error_msg = f"Failed to process query: {str(e)}"
                print(f"❌ Error: {error_msg}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": error_msg, 
                        "query": request.query,
                        "status": "error",
                        "timestamp": "2025-10-07"
                    }
                )

@app.post("/DatabaseBot")
async def db_query(request: QueryRequest):
    async with streamablehttp_client("http://localhost:10000/db/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Load tools and resources
            tools = await load_mcp_tools(session)
            schema = await session.read_resource("file://schema/candidates")
            # sql_prompt = (await session.get_prompt("sql_generator", {"user_query": request.query}))["text"]

            # # Build prompt context
            # user_query = "List candidates with CGPA > 8.5"
            full_prompt = f"""
            Database schema:
            {schema}

            SQL generator instructions:

            You are an SQL expert.
            Convert the following recruiter request into a valid MySQL SELECT query.
            Use only the provided table and columns.

            Table: candidates
            Columns: id, name, cgpa, skills, experience

            After generating SQL, immediately call the execute_sql tool to fetch results.
            Return only the query result, not the SQL text.

            User query:
            {request.query}
            """
            # 🔹 Setup Gemini LLM
            google_api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                max_retries=2,
                google_api_key=google_api_key
            )

            # 🔹 Create LangGraph Agent
            agent = create_react_agent(llm, tools)

            # Use the user's query
            query = [
                {"role": "user", "content": full_prompt}
            ]

            try:
                # 🔹 Invoke the agent
                print(f"🔍 Processing query: {full_prompt}")
                response = await agent.ainvoke({"messages": full_prompt})
                # print(response)

                # ✅ Extract only final textual response
                final_output = response["messages"][-1].content

                print("🤖 Final Output:", final_output)
                return {
                    "result": final_output, 
                    "query": full_prompt,
                    "status": "success"
                }
            
            except Exception as e:
                error_msg = f"Failed to process query: {str(e)}"
                print(f"❌ Error: {error_msg}")
                return {
                    "error": error_msg, 
                    "query": request.query,
                    "status": "error"
                }


@app.post("/comparer")
async def comparer_query(
    usernames: Annotated[str | None, Query(description="Comma-separated GitHub usernames")]
):
    """
    Compare multiple GitHub users using MCP data + Gemini LLM.
    Body: { "query": "custom comparison request" }
    Params: ?usernames=user1,user2,user3
    """

    # ✅ Step 1: Validate inputs
    if not usernames:
        return {"status": "error", "message": "Please provide usernames as query params, e.g. ?usernames=user1,user2"}

    username_list = [u.strip() for u in usernames.split(",") if u.strip()]
    if not username_list:
        return {"status": "error", "message": "No valid usernames found."}

    print("👥 GitHub Usernames:", username_list)
   

    # Step 2: Connect to MCP Server
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Load MCP tools
            tools = await load_mcp_tools(session)
            print("Loaded MCP Tools:", [t.name for t in tools])

            # # Find the GitHub proficiency tool
            # try:
            #     tool = next(t for t in tools if "GitHub Languages Proficiency" in t.description)
            # except StopIteration:
            #     return {"status": "error", "message": "MCP tool 'Get GitHub Languages Proficiency used by a user' not found."}

            # Step 3: Fetch GitHub data concurrently
            # async def fetch_user_data(username):
            #     try:
            #         print(f"📊 Fetching data for {username}...")
            #         result = await tool.ainvoke({"username": username})
            #         return username, result
            #     except Exception as e:
            #         print(f"⚠️ Error fetching {username}: {e}")
            #         return username, {"error": str(e)}

            # results = await asyncio.gather(*(fetch_user_data(u) for u in username_list))
            # user_data = dict(results)

            #  Step 4: Initialize Gemini LLM
            load_dotenv()
            google_api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                max_retries=2,
                google_api_key=google_api_key
            )

            #  Step 5: Create LangGraph Agent
            agent = create_react_agent(llm, tools)

            #  Step 6: Build prompt for the agent
            full_prompt = f"""
            Compare the following GitHub users.
            First fetch the data for each user using the 'Get GitHub Languages Proficiency used by a user' tool for each username.
            use only the tools to fetch the data don't make up any data on your own.
            your PAT_TOKEN is {os.getenv("PAT_TOKEN")}
            Users:
            {', '.join(username_list)}

            Generate a clear, structured comparison including:
            - Language proficiency & dominant languages
            - Repository and contribution insights
            - Strengths and specialization areas
            - Overall comparison summary
            give the final output in a more structured way with headings and subheadings as short summary report.
            """

            print("🧠 Sending prompt to Gemini...")

            # ✅ Step 7: Invoke the agent
            try:
                response = await agent.ainvoke({"messages": full_prompt})
                final_output = response["messages"][-1].content

                print("🤖 Final Output:", final_output)

                return {
                    "status": "success",
                    "query": full_prompt,
                    "usernames": username_list,
                    "result": final_output
                }

            except Exception as e:
                print(f"❌ LLM processing failed: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to process comparison: {str(e)}"
                }


@app.post("/comparerchatbot")
async def comparer_chatbot(
    request: QueryRequest,
    usernames: Annotated[str | None, Query(description="Comma-separated GitHub usernames")],
):
    """
    Smart Recruiter MCP Agent:
    Accepts a custom query and analyzes multiple GitHub profiles using MCP tools + Gemini.
    """

    # ✅ Step 1: Validate input
    if not usernames:
        return {"status": "error", "message": "Please provide usernames as query params, e.g. ?usernames=user1,user2"}

    username_list = [u.strip() for u in usernames.split(",") if u.strip()]
    if not username_list:
        return {"status": "error", "message": "No valid usernames found."}

    user_query = request.query
    print(f"👥 GitHub Users: {username_list}")
    print(f"💬 User Query: {user_query}")

    # ✅ Step 2: Connect to MCP Server
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ✅ Step 3: Load available MCP tools
            tools = await load_mcp_tools(session)
            print("🔧 Loaded MCP Tools:", [t.name for t in tools])

            # ✅ Step 4: Initialize Gemini LLM
            google_api_key = os.getenv("GOOGLE_API_KEY")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.2,
                max_retries=2,
                google_api_key=google_api_key
            )

            # ✅ Step 5: Create LangGraph Agent with MCP Tools
            agent = create_react_agent(llm, tools)

            # ✅ Step 6: Build the dynamic prompt for the agent
            full_prompt = f"""
            You are a Smart Recruiter Agent.

            You have access to the MCP tool: "Get GitHub Languages Proficiency used by a user".
            Use this tool to fetch data for each of the following GitHub users:
            {', '.join(username_list)}

            The personal access token (PAT) is available as: {os.getenv("PAT_TOKEN")}

            Then, based on their GitHub data, respond to the following recruiter query:
            "{user_query}"

            Instructions:
            - Always use MCP tools to gather data (never make up data)
            - Return a concise, well-structured report with sections and subheadings
            - If comparison is needed, highlight similarities and differences between users
            """

            print("🧠 Sending prompt to Gemini via MCP...")

            # ✅ Step 7: Invoke the MCP-integrated agent
            try:
                response = await agent.ainvoke({"messages": full_prompt})
                final_output = response["messages"][-1].content

                print("🤖 Final Output:", final_output)

                return {
                    "status": "success",
                    "query": user_query,
                    "usernames": username_list,
                    "result": final_output,
                }

            except Exception as e:
                print(f"❌ Error while processing: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to process recruiter query: {str(e)}",
                }

@app.get("/github/{username}/githubscore")
async def get_github_score(username: str, token: str = Query(None, description="GitHub Personal Access Token")):
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ✅ Step 3: Load available MCP tools
            tools = await load_mcp_tools(session)
            print("🔧 Loaded MCP Tools:", [t.name for t in tools])

                   # Pick the specific tool you want to call
            tool_name = "get_github_proficiency"  # Example tool name

            # Call the tool with required arguments
            response = await session.call_tool(
                tool_name,
                {"username": username,"token":token}  # your tool input params
            )
            # ✅ Access attributes properly (not subscripting)
            if response.isError:
                return {"error": "Tool returned an error", "details": response}

            # Extract text from first content item
            content_text = None
            if response.content and len(response.content) > 0:
                content_text = response.content[0].text

            # Try parsing JSON string
            try:
                json_data = json.loads(content_text)
                return json_data
            except Exception as e:
                return {
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_text": content_text
                }


@app.get("/codeprofiles/{username}/codeforcesscore")
async def get_codeforces_score(username: str):
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ✅ Step 3: Load available MCP tools
            tools = await load_mcp_tools(session)
            print("🔧 Loaded MCP Tools:", [t.name for t in tools])

                   # Pick the specific tool you want to call
            tool_name = "evaluate_codeforces_user"  # Example tool name

            # Call the tool with required arguments
            response = await session.call_tool(
                tool_name,
                {"codeforces_handle": username}  # your tool input params
            )
             # ✅ Access attributes properly (not subscripting)
            if response.isError:
                return {"error": "Tool returned an error", "details": response}

            # Extract text from first content item
            content_text = None
            if response.content and len(response.content) > 0:
                content_text = response.content[0].text

            # Try parsing JSON string
            try:
                json_data = json.loads(content_text)
                return json_data
            except Exception as e:
                return {
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_text": content_text
                }


@app.get("/leetcode/{username}")
async def get_leetcode_score(username: str):
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # ✅ Step 3: Load available MCP tools
            tools = await load_mcp_tools(session)
            print("🔧 Loaded MCP Tools:", [t.name for t in tools])

            # Pick the specific tool
            tool_name = "evaluate_leetcode_user"

            # Call the tool
            response = await session.call_tool(
                tool_name,
                {"username": username}
            )
            # Access attributes properly (not subscripting)
            if response.isError:
                return {"error": "Tool returned an error", "details": response}

            # Extract text from first content item
            content_text = None
            if response.content and len(response.content) > 0:
                content_text = response.content[0].text

            # Try parsing JSON string
            try:
                json_data = json.loads(content_text)
                return json_data
            except Exception as e:
                return {
                    "error": f"Failed to parse JSON: {str(e)}",
                    "raw_text": content_text
                }


@app.post("/evaluate/jdmatch")
async def evaluate_job_description(
    username: Annotated[str, Query(description="GitHub username or coding profile name")],
    job_description: str = Body(..., embed=True, description="Raw job description text"),
):
    """
    Evaluate how well a candidate's coding profile matches a given Job Description (JD).
    Uses MCP tools to fetch candidate data and Gemini to analyze tech stack compatibility.
    """

    # ✅ Step 1: Connect to MCP Server
    async with streamablehttp_client("http://localhost:10000/evaluate/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Load MCP tools
            tools = await load_mcp_tools(session)
            print("🧩 Loaded MCP Tools:", [t.name for t in tools])

            # ✅ Step 2: Initialize Gemini LLM
            load_dotenv()
            google_api_key = os.getenv("GOOGLE_API_KEY")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                max_retries=2,
                google_api_key=google_api_key,
            )

            # ✅ Step 3: Create a LangGraph Agent (can call MCP tools)
            agent = create_react_agent(llm, tools)

            # ✅ Step 4: Build prompt for the LLM
            full_prompt = f"""
            Your task is to identify the relevant technologies, programming languages, and tools from the following job description:

            {job_description}

            Instructions:
            1. If specific technologies, frameworks, or programming languages are mentioned directly (e.g., Python, React, Node.js), extract them exactly as stated.
            2. If broader fields or domains are mentioned (e.g., Machine Learning, Web Development, Data Science), infer the most relevant underlying tech stacks typically used in that domain.
            3. Return a concise, comma-separated list of languages only that need to be in github — no explanations or extra text.
            """

            print("📤 Sending JD comparison request to Gemini...")

            # ✅ Step 5b: Let the agent reason + use MCP tools (proper messages list)
            try:
                messages = [{"role": "user", "content": full_prompt}]
                response = await agent.ainvoke({"messages": messages})

                # Safely extract final text
                final_output = None
                if isinstance(response, dict) and response.get("messages"):
                    msgs = response.get("messages")
                    last = msgs[-1]
                    final_output = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
                if not final_output:
                    final_output = response.get("text") if isinstance(response, dict) else str(response)

                print(final_output)

                  # Pick the specific tool you want to call
                tool_name = "analyze_candidate"  # Example tool name

                # Call the tool with required arguments
                response = await session.call_tool(
                    tool_name,
                    {
                        "username": username, 
                        "job_description": final_output
                    }  
                )
                # ✅ Access attributes properly (not subscripting)
                if response.isError:
                    return {"error": "Tool returned an error", "details": response}

                # Extract text from first content item
                content_text = None
                if response.content and len(response.content) > 0:
                    content_text = response.content[0].text

                # Try parsing JSON string
                try:
                    json_data = json.loads(content_text)
                    return json_data
                except Exception as e:
                    return {
                        "error": f"Failed to parse JSON: {str(e)}",
                        "raw_text": content_text
                    }


            except Exception as e:
                error_msg = f"Failed to run agent: {str(e)}"
                print(f"❌ JD Evaluation failed: {error_msg}")
                return JSONResponse(status_code=500, content={
                    "status": "error",
                    "message": error_msg,
         
                })



client = MongoClient("mongodb://localhost:27017/")
db = client["studentDB"]
students_collection = db["students"]

class Student(BaseModel):
    name: str
    regno: str
    degree: str
    branch: str
    cgpa: float
    location: str
    expected_graduation_year: int
    university: str
    github_username: str
    leetcode_username: str
    codeforces_username: Optional[str] = None
    linkedin_username: str
    phone_number: str
    email:str
    resume_url: str

class StudentOut(BaseModel):
    id: str
    name: str
    regno: str
    degree: str
    branch: str
    cgpa: float
    location: str
    expected_graduation_year: int
    university: str
    github_username: str
    leetcode_username: str
    codeforces_username: Optional[str] = None
    linkedin_username: str
    phone_number: str
    email:str
    resume_url: str

def student_serializer(student) -> dict:
    return {
        "id": str(student["_id"]),
        "name": student["name"],
        "regno": student["regno"],
        "degree": student["degree"],
        "branch": student["branch"],
        "cgpa": student["cgpa"],
        "location": student["location"],
        "expected_graduation_year": student["expected_graduation_year"],
        "university": student["university"],
        "github_username": student["github_username"],
        "leetcode_username": student["leetcode_username"],
        "codeforces_username": student.get("codeforces_username"),
        "linkedin_username": student["linkedin_username"],
        "phone_number": student["phone_number"],
        "email": student["email"],
        "resume_url": student["resume_url"]
    }

@app.post("/students")
def create_student(student: Student):
    result = students_collection.insert_one(student.dict())
    return {"id": str(result.inserted_id)}

@app.get("/students", response_model=List[StudentOut])
def get_students():
    students = students_collection.find()
    return [student_serializer(s) for s in students]

@app.get("/students/{student_id}", response_model=StudentOut)
def get_student(student_id: str):
    student = students_collection.find_one({"_id": ObjectId(student_id)})
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student_serializer(student)

@app.put("/students/{student_id}")
def update_student(student_id: str, student: Student):
    result = students_collection.update_one({"_id": ObjectId(student_id)}, {"$set": student.dict()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student updated successfully"}

@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    result = students_collection.delete_one({"_id": ObjectId(student_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"message": "Student deleted successfully"}


# ------------------------------
# Pydantic Models
# ------------------------------
recruitment_db = client["recruitmentDB"]
job_applicants_collection = recruitment_db["jobapplicants"]

# ------------------------------
# Models
# ------------------------------
class GithubActivity(BaseModel):
    total_repos: Optional[int]
    total_stars: Optional[int]
    total_forks: Optional[int]
    total_commits: Optional[int]

class Github(BaseModel):
    username: Optional[str]
    job_description_preview: Optional[str]
    languages: Optional[Dict[str, int]]
    matched_keywords: Optional[List[str]]
    github_activity: Optional[GithubActivity]
    commit_factor: Optional[float]
    match_score: Optional[float]
    compatibility_score: Optional[float]

class LeetCodeProblems(BaseModel):
    Total: Optional[int]
    Easy: Optional[int]
    Medium: Optional[int]
    Hard: Optional[int]
    AcceptanceRate: Optional[str]

class LeetCodeContest(BaseModel):
    AttendedContests: Optional[int]

class LeetCodeBadge(BaseModel):
    id: Optional[str]
    name: Optional[str]
    icon: Optional[str]
    earnedOn: Optional[str]

class LeetCode(BaseModel):
    username: Optional[str]
    country: Optional[str]
    ranking: Optional[int]
    problemsSolved: Optional[LeetCodeProblems]
    contestStats: Optional[LeetCodeContest]
    badges: Optional[List[LeetCodeBadge]]

class CodeforcesDetails(BaseModel):
    lastName: Optional[str]
    country: Optional[str]
    city: Optional[str]
    rating: Optional[int]
    rank: Optional[str]
    maxRating: Optional[int]
    maxRank: Optional[str]

class Codeforces(BaseModel):
    handle: Optional[str]
    codeforces_details: Optional[CodeforcesDetails]

class JobApplicant(BaseModel):
    jobId: int
    studentId: int
    github: Optional[Github]
    leetcode: Optional[LeetCode]
    codeforces: Optional[Codeforces]

# ------------------------------
# Serializer
# ------------------------------
def applicant_serializer(applicant) -> dict:
    return {
        "id": str(applicant["_id"]),
        "jobId": applicant["jobId"],
        "studentId": applicant["studentId"],
        "github": applicant.get("github"),
        "leetcode": applicant.get("leetcode"),
        "codeforces": applicant.get("codeforces"),
    }

# ------------------------------
# CRUD Endpoints
# ------------------------------

@app.post("/jobapplicants")
def create_job_applicant(applicant: JobApplicant):
    """Create a new job applicant"""
    result = job_applicants_collection.insert_one(applicant.dict())
    return {"id": str(result.inserted_id), "message": "Job applicant created successfully"}

@app.get("/jobapplicants", response_model=List[dict])
def get_all_job_applicants():
    """Get all job applicants"""
    applicants = job_applicants_collection.find()
    return [applicant_serializer(a) for a in applicants]

@app.get("/jobapplicants/{applicant_id}")
def get_job_applicant(applicant_id: str):
    """Get a specific job applicant"""
    applicant = job_applicants_collection.find_one({"_id": ObjectId(applicant_id)})
    if not applicant:
        raise HTTPException(status_code=404, detail="Job applicant not found")
    return applicant_serializer(applicant)

@app.put("/jobapplicants/{applicant_id}")
def update_job_applicant(applicant_id: str, applicant: JobApplicant):
    """Update job applicant details"""
    result = job_applicants_collection.update_one(
        {"_id": ObjectId(applicant_id)}, {"$set": applicant.dict()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job applicant not found")
    return {"message": "Job applicant updated successfully"}

@app.delete("/jobapplicants/{applicant_id}")
def delete_job_applicant(applicant_id: str):
    """Delete job applicant"""
    result = job_applicants_collection.delete_one({"_id": ObjectId(applicant_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job applicant not found")
    return {"message": "Job applicant deleted successfully"}
class GithubActivity(BaseModel):
    total_repos: Optional[int]
    total_stars: Optional[int]
    total_forks: Optional[int]
    total_commits: Optional[int]

class Github(BaseModel):
    username: Optional[str]
    job_description_preview: Optional[str]
    languages: Optional[Dict[str, int]]
    matched_keywords: Optional[List[str]]
    github_activity: Optional[GithubActivity]
    commit_factor: Optional[float]
    match_score: Optional[float]
    compatibility_score: Optional[float]

class LeetcodeProblems(BaseModel):
    Total: Optional[int]
    Easy: Optional[int]
    Medium: Optional[int]
    Hard: Optional[int]
    AcceptanceRate: Optional[str]

class LeetcodeContestStats(BaseModel):
    AttendedContests: Optional[int]

class LeetcodeBadge(BaseModel):
    id: Optional[str]
    name: Optional[str]
    icon: Optional[str]
    earnedOn: Optional[str]

class Leetcode(BaseModel):
    username: Optional[str]
    country: Optional[str]
    ranking: Optional[int]
    problemsSolved: Optional[LeetcodeProblems]
    contestStats: Optional[LeetcodeContestStats]
    badges: Optional[List[LeetcodeBadge]]

class CodeforcesDetails(BaseModel):
    lastName: Optional[str]
    country: Optional[str]
    city: Optional[str]
    rating: Optional[int]
    rank: Optional[str]
    maxRating: Optional[int]
    maxRank: Optional[str]

class Codeforces(BaseModel):
    handle: Optional[str]
    codeforces_details: Optional[CodeforcesDetails]

class JobApplicant(BaseModel):
    jobId: int = Field(..., description="Job ID reference")
    studentId: int = Field(..., description="Student ID reference")
    github: Optional[Github]
    leetcode: Optional[Leetcode]
    codeforces: Optional[Codeforces]

# Helper: convert ObjectId
def serialize_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

jobApplicant_db = client["recruitmentDB"]
jobs_collectionthree = jobApplicant_db ["jobapplicants"]

# CREATE
@app.post("/jobapplicants", response_model=JobApplicant)
def create_job_applicant(applicant: JobApplicant):
    result = jobs_collectionthree.insert_one(applicant.dict())
    new_doc =jobs_collectionthree.find_one({"_id": result.inserted_id})
    return serialize_doc(new_doc)

# READ ALL
@app.get("/jobapplicants")
def get_all_job_applicants():
    docs = jobs_collectionthree.find().to_list(100)
    return [serialize_doc(doc) for doc in docs]

# READ ONE
@app.get("/jobapplicants/{id}")
def get_job_applicant(id: str):
    doc = jobs_collectionthree.find_one({"_id": ObjectId(id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return serialize_doc(doc)

# UPDATE
@app.put("/jobapplicants/{id}")
def update_job_applicant(id: str, applicant: JobApplicant):
    update_data = {k: v for k, v in applicant.dict().items() if v is not None}
    result = jobs_collectionthree.update_one({"_id": ObjectId(id)}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Applicant not found")
    updated = jobs_collectionthree.find_one({"_id": ObjectId(id)})
    return serialize_doc(updated)

# DELETE
@app.delete("/jobapplicants/{id}")
def delete_job_applicant(id: str):
    result = jobs_collectionthree.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return {"message": "Applicant deleted successfully"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)