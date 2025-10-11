# app.py
import json
import asyncio
import os
import httpx
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Annotated
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


@app.get("/codeprofiles/{username}/leetcodescore")
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
                {"leetcode_username": username}
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
            You are an AI technical recruiter. Analyze the following job description and extract
            all the core technologies, frameworks, and required skills from it.

            Then, using the MCP tools provided, fetch the GitHub or coding profile data for the user '{username}'.
            use the tool generative `get_github_proficiency` to fetch the user's github details.
            use the PAT_Token {os.getenv("PAT_TOKEN")} to fetch the data.
            Compare the extracted JD skill sets with the user's skills, repositories, and technologies.

            Output a detailed JSON response with:
            - extracted_jd_techs: [list of technologies in JD]
            - candidate_techs: [list of technologies user actually uses]
            - compatibility_score: number from 0 to 100 (how well the candidate matches)
            - reasoning: short paragraph explaining the compatibility level

            Job Description:
            {job_description}

            Username:
            {username}
            """

            print("📤 Sending JD comparison request to Gemini...")

            # ✅ Step 5: Let the agent reason + use MCP tools
            try:
                response = await agent.ainvoke({"messages": full_prompt})
                final_output = response["messages"][-1].content

                print("✅ JD–Resume Match Result:", final_output)

                return {
                    "status": "success",
                    "username": username,
                    "result": final_output,
                }

            except Exception as e:
                print(f"❌ JD Evaluation failed: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to evaluate JD compatibility: {str(e)}",
                }

from typing import Optional, List

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)