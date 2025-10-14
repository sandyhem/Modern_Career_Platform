# app.py
import json
import asyncio
import os
import httpx
from datetime import datetime

from fastapi import Body, FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from typing import Annotated
from typing import Dict, List, Optional

from requests import session
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException
from bson import ObjectId
from pymongo import MongoClient

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


from fastapi import File, UploadFile, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import io
import google.generativeai as genai
import PyPDF2
# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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
        "http://localhost:5173",
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


import re
import ast



# ------------------ Helper function ------------------
def clean_llm_output(text: str) -> dict:
    """
    Convert LLM output (possibly wrapped in Markdown) into a Python dict.
    """
    # Remove code fences
    text = re.sub(r"```(?:python)?\n?", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        # Safely parse dict from string
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Cannot convert LLM output to dict: {e}\nOutput: {text}")

# ------------------ Endpoint ------------------
@app.post("/mongoreader/")
async def mongodb_query(request: QueryRequest):
    user_query = request.query
    print("Query from body:", user_query)

    # Step 1: Connect to MCP Server
    async with streamablehttp_client("http://localhost:10000/db/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Load MCP tools, prompts, resources
            tools = await load_mcp_tools(session)
            prompts = await session.list_prompts()
            resources = await session.list_resources()

            # Load schema resource
            resource_name = "schema://jobapplicants"
            resource_content = await session.read_resource(resource_name)
            schema_text = resource_content.contents[0].text
            print(f"\nResource '{resource_name}' schema JSON:\n", schema_text)

            # Load translation prompt
            prompt_name = "translate_query_prompt"
            prompt_content = await session.get_prompt(prompt_name)
            prompt_text = prompt_content.messages[0].content.text
            print(f"\nPrompt '{prompt_name}' text:\n", prompt_text)

            # Step 2: Initialize Gemini LLM
            load_dotenv()
            google_api_key = os.getenv("GOOGLE_API_KEY")

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                max_retries=2,
                google_api_key=google_api_key,
            )

            # Step 3: Create LangGraph Agent
            agent = create_react_agent(llm, tools)

            # Step 4: Build full prompt
            full_prompt = f"""
            schema:
            {schema_text}
            query:
            {user_query}
            prompt:
            You are a MongoDB query translator. Your task is to convert natural language questions into valid MongoDB query filters (Python dict format).
            **SCHEMA REFERENCE:**
            Use the schema from resource to understand available fields and their types.

            **TRANSLATION RULES:**
            1. Output ONLY a valid Python dictionary that can be used with pymongo's find() method
            2. Use MongoDB query operators: $gt, $lt, $gte, $lte, $eq, $ne, $in, $nin, $and, $or, $regex, $exists
            3. For text searches, use case-insensitive regex: {{"field": {{"$regex": "pattern", "$options": "i"}}}}
            4. For array fields (like skills), use $in or $all operators
            5. Combine multiple conditions using $and or $or
            6. For date comparisons, use ISO format strings or ISODate syntax
            7. Return empty dict for queries requesting all documents

            **EXAMPLES:**

            Query: "Find applicants with github score above 0.8"
            Output: {{"github_match_score": {{"$gt": 0.8}}}}

            Query: "Show me Python developers in pending status"
            Output: {{"$and": [{{"skills": {{"$regex": "python", "$options": "i"}}}}, {{"status": "pending"}}]}}

            Query: "Applicants with more than 5 years experience and overall score >= 0.7"
            Output: {{"$and": [{{"experience_years": {{"$gt": 5}}}}, {{"overall_score": {{"$gte": 0.7}}}}]}}

            Query: "Find candidates who applied for Senior Developer or Tech Lead positions"
            Output: {{"applied_position": {{"$in": ["Senior Developer", "Tech Lead"]}}}}

            **YOUR TASK:**
            Translate the following natural language query into a MongoDB filter dictionary:
            After generating filter, immediately call the tool `execute_query` with description "Execute a MongoDB query and return results" as output.
            Use the query result and give the enhanced response to the user.
            """

            print("📤 Sending query to Gemini...")

            try:
                # Step 5: Ask LLM to translate query
                messages = [{"role": "user", "content": full_prompt}]
                response = await agent.ainvoke({"messages": messages})

                # Extract final text from agent
                final_output = None
                if isinstance(response, dict) and response.get("messages"):
                    last_msg = response["messages"][-1]
                    final_output = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else None)
                if not final_output:
                    final_output = response.get("text") if isinstance(response, dict) else str(response)

                print("LLM output:\n", final_output)

                return final_output

            except Exception as e:
                error_msg = f"Failed to run agent: {str(e)}"
                print(f"❌ MongoDB Query failed: {error_msg}")
                return JSONResponse(status_code=500, content={
                    "status": "error",
                    "message": error_msg
                })
# ------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["studentDB"]
register_students_collection=db["auth_student"]

SECRET_KEY = os.getenv('SECRET_KEY','defaultsecretkey')
ALGORITHM = os.getenv('ALGORITHM','HS256')
EXPIRY = os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES',60)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

class RegisterStudent(BaseModel):
    username:str
    email:str
    password:str
    usertype:str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict
    
class Student(BaseModel):
    sid:str
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
    sid: str
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
        "sid": str(student["sid"]),  
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


def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=EXPIRY)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_student_by_email(email: str):
    return register_students_collection.find_one({"email": email})

def authenticate_student(email: str, password: str):
    student = get_student_by_email(email)
    if not student:
        return False
    if not verify_password(password, student["password"]):
        return False
    return student

# ---------------- Signup ----------------
@app.post("/signup",tags=["auth"])
def signup(student: RegisterStudent):
    if get_student_by_email(student.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    #print(student.password)
    hashed_password = hash_password(student.password)
    student_dict = student.model_dump()
    student_dict["password"] = hashed_password
    register_students_collection.insert_one(student_dict)
    access_token = create_access_token(data={"sub": student.email})
    student_details=get_student_by_email(student.email)
    if "_id" in student_details:
        student_details["_id"] = str(student_details["_id"]) 
    return {"message": "Student registered successfully","access_token":access_token,"user":student_details}

# ---------------- Login ----------------
class LoginRequest(BaseModel):
    username:str
    password:str
    
@app.post("/login", response_model=Token,tags=["auth"])
def login(form_data: LoginRequest):
    student = authenticate_student(form_data.username, form_data.password)
    if not student:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    student_details=get_student_by_email(form_data.username)
    print(student_details)
    if "_id" in student_details:
        student_details["_id"] = str(student_details["_id"]) 
    access_token = create_access_token(data={"sub": student["email"]})
    return {"access_token": access_token, "token_type": "bearer", "user": student_details}

# ---------------- Protected Route Example ----------------
@app.get("/profile")
def read_profile(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    student = get_student_by_email(email)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    return student

students_collection = db["students"]

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
    # Try to fetch by MongoDB ObjectId first
    try:
        object_id = ObjectId(student_id)
        student = students_collection.find_one({"_id": object_id})
        if student:
            return student_serializer(student)
    except Exception:
        pass  # Not a valid ObjectId, try by sid

    # Fallback: fetch by sid field
    student = students_collection.find_one({"sid": student_id})
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


class LeetCodeBadge(BaseModel):
    id: Optional[str]
    name: Optional[str]
    icon: Optional[str]
    earnedOn: Optional[str]

class JobApplicant(BaseModel):
    jobId: str
    studentId: str
    github_username: Optional[str]
    github_job_description_preview: Optional[str]
    github_languages: Optional[Dict[str, int]]
    github_matched_keywords: Optional[List[str]]
    github_commit_factor: Optional[float]
    github_match_score: Optional[float]
    github_compatibility_score: Optional[float]
    total_repos: Optional[int]
    total_stars: Optional[int]
    total_forks: Optional[int]
    total_commits: Optional[int]

    leetcode_username: Optional[str]
    leetcode_country: Optional[str]
    leetcode_ranking: Optional[int]
    leetcode_totalproblems: Optional[int]
    leetcode_easy: Optional[int]
    leetcode_medium: Optional[int]
    leetcode_hard: Optional[int]
    leetcode_acceptanceRate: Optional[str]
    leetcode_contests: Optional[int]
    badges: Optional[List[LeetCodeBadge]]

    codeforces_handle: Optional[str]
    codeforces_rating: Optional[int]
    codeforces_rank: Optional[str]
    codeforces_maxRating: Optional[int]
    codeforces_maxRank: Optional[str]

# ------------------------------
# Serializer
# ------------------------------
def applicant_serializer(applicant) -> dict:
    return {
        "id": str(applicant["_id"]),
        "jobId": applicant.get("jobId"),
        "studentId": applicant.get("studentId"),

        # GitHub fields
        "github_username": applicant.get("github_username"),
        "github_job_description_preview": applicant.get("github_job_description_preview"),
        "github_languages": applicant.get("github_languages"),
        "github_matched_keywords": applicant.get("github_matched_keywords"),
        "github_commit_factor": applicant.get("github_commit_factor"),
        "github_match_score": applicant.get("github_match_score"),
        "github_compatibility_score": applicant.get("github_compatibility_score"),
        "total_repos": applicant.get("total_repos"),
        "total_stars": applicant.get("total_stars"),
        "total_forks": applicant.get("total_forks"),
        "total_commits": applicant.get("total_commits"),

        # LeetCode fields
        "leetcode_username": applicant.get("leetcode_username"),
        "leetcode_country": applicant.get("leetcode_country"),
        "leetcode_ranking": applicant.get("leetcode_ranking"),
        "leetcode_totalproblems": applicant.get("leetcode_totalproblems"),
        "leetcode_easy": applicant.get("leetcode_easy"),
        "leetcode_medium": applicant.get("leetcode_medium"),
        "leetcode_hard": applicant.get("leetcode_hard"),
        "leetcode_acceptanceRate": applicant.get("leetcode_acceptanceRate"),
        "leetcode_contests": applicant.get("leetcode_contests"),
        "badges": applicant.get("badges"),

        # Codeforces fields
        "codeforces_handle": applicant.get("codeforces_handle"),
        "codeforces_rating": applicant.get("codeforces_rating"),
        "codeforces_rank": applicant.get("codeforces_rank"),
        "codeforces_maxRating": applicant.get("codeforces_maxRating"),
        "codeforces_maxRank": applicant.get("codeforces_maxRank"),
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

@app.get("/jobapplicants/{job_id}")
def get_job_applicants_by_jobid(job_id: str):
    """Get all job applicants for a specific jobId"""
    applicants = job_applicants_collection.find({"jobId": job_id})
    result = [applicant_serializer(a) for a in applicants]
    if not result:
        raise HTTPException(status_code=404, detail="No applicants found for this jobId")
    return result

@app.get("/jobapplicants/student/{student_id}")
def get_job_applicants_by_studentid(student_id: str):
    """Get all job applicants for a specific studentId"""
    applicants = job_applicants_collection.find({"studentId": student_id})
    result = [applicant_serializer(a) for a in applicants]
    if not result:
        raise HTTPException(status_code=404, detail="No applicants found for this studentId")
    return result

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

# ------------------------------
# Pydantic Models
# ------------------------------
recruitment_db = client["recruitmentDB"]
jobs_collectiontwo = recruitment_db["jobs"]

class Job(BaseModel):
    title: str
    type: str = Field(default="Full-Time")
    department: str
    company_name:str
    location: str
    postDate: datetime = Field(default_factory=datetime.utcnow)
    endDate: Optional[datetime] = None
    responsibilities: str
    qualifications: str
    skills: str

class JobOut(BaseModel):
    id: str
    title: str
    type: str
    department: str
    company_name: str
    location: str
    postDate: datetime
    endDate: Optional[datetime]
    responsibilities: str
    qualifications: str
    skills: str

# ------------------------------
# Serializer
# ------------------------------

def job_serializer(job) -> dict:
    return {
        "id": str(job["_id"]),
        "title": job["title"],
        "type": job["type"],
        "department": job["department"],
        "location": job["location"],
        "company_name": job["company_name"],
        "postDate": job["postDate"],
        "endDate": job.get("endDate"),
        "responsibilities": job["responsibilities"],
        "qualifications": job["qualifications"],
        "skills": job["skills"],
    }

# ------------------------------
# CRUD Endpoints
# ------------------------------

@app.post("/jobs")
def create_job(job: Job):
    """Create a new job post"""
    result = jobs_collectiontwo.insert_one(job.dict())
    return {"id": str(result.inserted_id), "message": "Job created successfully"}

@app.get("/jobs", response_model=List[JobOut])
def get_jobs():
    """Get all job posts"""
    jobs = jobs_collectiontwo.find()
    return [job_serializer(j) for j in jobs]

@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    """Get a specific job by ID"""
    job = jobs_collectiontwo.find_one({"_id": ObjectId(job_id)})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_serializer(job)

@app.put("/jobs/{job_id}")
def update_job(job_id: str, job: Job):
    """Update an existing job"""
    result = jobs_collectiontwo.update_one({"_id": ObjectId(job_id)}, {"$set": job.dict()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job updated successfully"}

@app.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job"""
    result = jobs_collectiontwo.delete_one({"_id": ObjectId(job_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}

"""
university_server.py
Backend server for University Student Verification Portal
Handles Excel upload and stores data in MySQL database
"""

# ==================== DATABASE SETUP ====================
DATABASE_URL = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://root:root@localhost:3306/universityDB"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== DATABASE MODEL ====================
class StudentVerificationDB(Base):
    __tablename__ = "student_verifications"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    register_number = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    year_of_study = Column(Integer, nullable=False)
    cgpa = Column(Float, nullable=False)
    total_backlogs = Column(Integer, default=0)
    current_arrears = Column(Integer, default=0)
    history_of_arrears = Column(Boolean, default=False)
    placement_eligible = Column(Boolean, default=False)
    internships = Column(Text, nullable=True)
    verification_status = Column(String(20), default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC MODELS ====================
class StudentVerification(BaseModel):
    register_number: str
    name: str
    department: str
    year_of_study: int
    cgpa: float
    total_backlogs: Optional[int] = 0
    current_arrears: Optional[int] = 0
    history_of_arrears: bool
    placement_eligible: bool
    internships: Optional[List[str]] = []
    verification_status: str = "Pending"

class StudentVerificationResponse(StudentVerification):
    id: int
    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    success: bool
    message: str
    records_added: int
    records_updated: int
    errors: List[str] = []

class UpdateStatusRequest(BaseModel):
    verification_status: str

# ==================== DATABASE DEPENDENCY ====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== HELPER ====================
def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes"]
    if isinstance(value, (int, float)):
        return bool(value)
    return False

# ==================== ENDPOINTS ====================

@app.post("/upload-excel", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")

    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        required_columns = [
            'register_number', 'name', 'department', 'year_of_study', 'cgpa',
            'total_backlogs', 'current_arrears', 'history_of_arrears',
            'placement_eligible', 'internships', 'verification_status'
        ]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

        db = next(get_db())
        added, updated = 0, 0
        errors = []

        for i, row in df.iterrows():
            try:
                internships = str(row.get("internships", "")).strip()
                internships_list = [x.strip() for x in internships.split(",") if x.strip() and x.lower() != "none"]

                existing = db.query(StudentVerificationDB).filter_by(register_number=str(row["register_number"])).first()
                if existing:
                    existing.name = str(row["name"])
                    existing.department = str(row["department"])
                    existing.year_of_study = int(row["year_of_study"])
                    existing.cgpa = float(row["cgpa"])
                    existing.total_backlogs = int(row.get("total_backlogs", 0))
                    existing.current_arrears = int(row.get("current_arrears", 0))
                    existing.history_of_arrears = to_bool(row.get("history_of_arrears", False))
                    existing.placement_eligible = to_bool(row.get("placement_eligible", False))
                    existing.internships = ",".join(internships_list) if internships_list else None
                    existing.verification_status = str(row.get("verification_status", "Pending"))
                    updated += 1
                else:
                    new_stud = StudentVerificationDB(
                        register_number=str(row["register_number"]),
                        name=str(row["name"]),
                        department=str(row["department"]),
                        year_of_study=int(row["year_of_study"]),
                        cgpa=float(row["cgpa"]),
                        total_backlogs=int(row.get("total_backlogs", 0)),
                        current_arrears=int(row.get("current_arrears", 0)),
                        history_of_arrears=to_bool(row.get("history_of_arrears", False)),
                        placement_eligible=to_bool(row.get("placement_eligible", False)),
                        internships=",".join(internships_list) if internships_list else None,
                        verification_status=str(row.get("verification_status", "Pending")),
                    )
                    db.add(new_stud)
                    added += 1

            except Exception as e:
                errors.append(f"Row {i+2}: {str(e)}")

        db.commit()

        return UploadResponse(
            success=True,
            message="Excel uploaded successfully",
            records_added=added,
            records_updated=updated,
            errors=errors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/students-details", response_model=List[StudentVerificationResponse])
def get_students():
    db = next(get_db())
    students = db.query(StudentVerificationDB).all()
    result = []
    for s in students:
        result.append(StudentVerificationResponse(
            id=s.id,
            register_number=s.register_number,
            name=s.name,
            department=s.department,
            year_of_study=s.year_of_study,
            cgpa=s.cgpa,
            total_backlogs=s.total_backlogs,
            current_arrears=s.current_arrears,
            history_of_arrears=s.history_of_arrears,
            placement_eligible=s.placement_eligible,
            internships=s.internships.split(",") if s.internships else [],
            verification_status=s.verification_status,
        ))
    return result

@app.put("/students/{register_number}/status")
def update_status(register_number: str, update: UpdateStatusRequest):
    db = next(get_db())
    student = db.query(StudentVerificationDB).filter_by(register_number=register_number).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.verification_status = update.verification_status
    db.commit()
    return {"message": f"Status updated to {update.verification_status}"}

@app.get("/stats")
def get_statistics():
    db = next(get_db())

    total_students = db.query(StudentVerificationDB).count()
    verified = db.query(StudentVerificationDB).filter_by(verification_status="Verified").count()
    pending = db.query(StudentVerificationDB).filter_by(verification_status="Pending").count()
    rejected = db.query(StudentVerificationDB).filter_by(verification_status="Rejected").count()
    placement_eligible = db.query(StudentVerificationDB).filter_by(placement_eligible=True).count()

    # Department-wise count
    departments = {}
    department_data = db.query(StudentVerificationDB.department).all()
    for (dept,) in department_data:
        if dept in departments:
            departments[dept] += 1
        else:
            departments[dept] = 1

    return {
        "total_students": total_students,
        "verified": verified,
        "pending": pending,
        "rejected": rejected,
        "placement_eligible": placement_eligible,
        "departments": departments
    }


# ==================== MODELS ====================
class ResumeTextResponse(BaseModel):
    text: str
    filename: str

class ResumeAnalysisRequest(BaseModel):
    resume_text: str

class ResumeAnalysisResponse(BaseModel):
    keywords: List[str]
    skills: List[str]
    experience: str
    summary: str
    education: Optional[str] = None
    certifications: Optional[List[str]] = None

# ==================== HELPER FUNCTIONS ====================
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Word document: {str(e)}")

def parse_gemini_response(response_text: str) -> dict:
    """Parse Gemini's response and extract structured data"""
    try:
        # Try to parse as JSON first
        if "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        
        # Fallback: Extract using regex patterns
        data = {
            "keywords": [],
            "skills": [],
            "experience": "",
            "summary": "",
            "education": "",
            "certifications": []
        }
        
        # Extract keywords
        keywords_match = re.search(r'(?:keywords?|key\s*words?):\s*\[?([^\]]+)\]?', response_text, re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1)
            data["keywords"] = [k.strip().strip('"\'') for k in keywords_text.split(',')]
        
        # Extract skills
        skills_match = re.search(r'(?:skills?):\s*\[?([^\]]+)\]?', response_text, re.IGNORECASE)
        if skills_match:
            skills_text = skills_match.group(1)
            data["skills"] = [s.strip().strip('"\'') for s in skills_text.split(',')]
        
        # Extract experience
        exp_match = re.search(r'(?:experience|experience\s*level):\s*["\']?([^"\'\\n]+)["\']?', response_text, re.IGNORECASE)
        if exp_match:
            data["experience"] = exp_match.group(1).strip()
        
        # Extract summary
        summary_match = re.search(r'(?:summary|professional\s*summary):\s*["\']?([^"\']+)["\']?', response_text, re.IGNORECASE)
        if summary_match:
            data["summary"] = summary_match.group(1).strip()
        
        return data
        
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        # Return basic structure with whatever we can extract
        return {
            "keywords": [],
            "skills": [],
            "experience": "Not specified",
            "summary": response_text[:200] if response_text else "",
            "education": "",
            "certifications": []
        }

# ==================== ENDPOINTS ====================

@app.post("/extract-resume-text", response_model=ResumeTextResponse)
async def extract_resume_text(resume: UploadFile = File(...)):
    """
    Extract text content from uploaded resume (PDF or Word)
    """
    if not resume.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file type
    file_extension = resume.filename.lower().split('.')[-1]
    
    if file_extension not in ['pdf', 'doc', 'docx']:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and Word documents are supported"
        )
    
    try:
        # Read file content
        file_content = await resume.read()
        
        # Extract text based on file type
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension in ['doc', 'docx']:
            text = extract_text_from_docx(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        if not text or len(text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from resume. Please ensure the file is readable."
            )
        
        return ResumeTextResponse(
            text=text,
            filename=resume.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )


@app.post("/analyze-resume-gemini", response_model=ResumeAnalysisResponse)
async def analyze_resume_with_gemini(request: ResumeAnalysisRequest):
    """
    Analyze resume text using Gemini AI to extract keywords, skills, and experience
    """
    if not request.resume_text or len(request.resume_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Resume text is too short or empty"
        )
    
    try:
        # Create Gemini prompt
        prompt = f"""
        You are an expert resume analyzer and career advisor. Analyze the following resume text and extract key information.
        
        RESUME TEXT:
        {request.resume_text}
        
        Please provide a structured JSON response with the following fields:
        
        1. "keywords": A list of important keywords and technical terms found in the resume (15-25 keywords)
        2. "skills": A list of technical skills, tools, and technologies (programming languages, frameworks, software, etc.)
        3. "experience": A brief summary of the candidate's experience level (e.g., "Entry Level", "2-3 years", "Senior with 5+ years", etc.)
        4. "summary": A brief professional summary of the candidate (2-3 sentences)
        5. "education": Educational background (degree, institution, year)
        6. "certifications": List of certifications or achievements mentioned
        
        Format your response as valid JSON that can be parsed. Be concise and focus on extracting actionable information for job matching.
        
        Example format:
        {{
          "keywords": ["Python", "Machine Learning", "Data Analysis", "SQL", "AWS"],
          "skills": ["Python", "TensorFlow", "Pandas", "Docker", "Git"],
          "experience": "Mid-level with 3-4 years of experience in data science",
          "summary": "Data scientist with strong background in machine learning and statistical analysis. Experienced in building predictive models and working with large datasets.",
          "education": "Master's in Computer Science, Stanford University, 2020",
          "certifications": ["AWS Certified Solutions Architect", "Google Data Analytics Professional"]
        }}
        """
        
        # Call Gemini API
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            raise HTTPException(
                status_code=500,
                detail="Gemini API returned empty response"
            )
        
        # Parse Gemini response
        parsed_data = parse_gemini_response(response.text)
        
        # Ensure we have at least some data
        if not parsed_data.get("keywords") and not parsed_data.get("skills"):
            # Fallback: Extract basic keywords from resume text
            words = request.resume_text.lower().split()
            common_skills = ['python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 
                           'docker', 'kubernetes', 'git', 'machine learning', 'data science']
            parsed_data["keywords"] = [skill for skill in common_skills if skill in ' '.join(words)]
            parsed_data["skills"] = parsed_data["keywords"][:10]
        
        return ResumeAnalysisResponse(
            keywords=parsed_data.get("keywords", [])[:25],
            skills=parsed_data.get("skills", [])[:20],
            experience=parsed_data.get("experience", "Not specified"),
            summary=parsed_data.get("summary", ""),
            education=parsed_data.get("education"),
            certifications=parsed_data.get("certifications", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing resume with Gemini: {str(e)}"
        )


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# This file contains the core logic to send emails to students

def send_student_mail(receiver_email: str, selected: bool, feedback: str = ""):
    """
    Sends an email to the student notifying about their selection status.
    
    Args:
        receiver_email (str): Student's email address
        selected (bool): True if selected, False if not selected
        feedback (str): Optional feedback if not selected
    """
    # 💡 Replace these with real credentials or use environment variables
    SENDER_EMAIL = "harishma.k675@gmail.com"
    SENDER_PASSWORD = "haif cnfx gapb jswc"  # use App Password if Gmail 2FA is on

    # ✉ Define mail subject & body based on selection status
    if selected:
        subject = "🎉 Congratulations! You've been Selected"
        body = f"""
        Dear Student,

        Congratulations! We are thrilled to inform you that you have been selected for the next stage.

        Keep up your great work and prepare for the onboarding process.

        Best Regards,  
        Recruitment Team
        """
    else:
        subject = "Feedback on Your Application"
        body = f"""
        Dear Student,

        Thank you for applying. Although you were not selected this time, we appreciate your effort.
        
        Feedback: {feedback}

        Keep improving, and we encourage you to apply again in the future.

        Best Regards,  
        Recruitment Team
        """

    # 🏗 Create the email message
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        # 💌 Connect to Gmail SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"✅ Mail sent successfully to {receiver_email}")
        return {"status": "success", "message": "Mail sent successfully"}
    except Exception as e:
        print(f"❌ Error sending mail: {e}")
        return {"status": "error", "message": str(e)}
                
class MailRequest(BaseModel):
    sid: str
    email: str
    selected: bool
    feedback: str | None = None

def jobdata(student_id: str):
    """Get all job applicants for a specific studentId"""
    applicants = job_applicants_collection.find({"studentId": student_id})
    result = [applicant_serializer(a) for a in applicants]
    if not result:
        raise HTTPException(status_code=404, detail="No applicants found for this studentId")
    return result

@app.post("/notify_student", tags=["Recruiter"])
async def notify_student(request: MailRequest):
    """
    Recruiter endpoint to send selection or feedback email.
    """
    # Step 2: Initialize Gemini LLM
    """Get all job applicants for a specific studentId"""
    print(request.sid)
    applicants=jobdata(request.sid)
    print(applicants)

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        max_retries=2,
        google_api_key=google_api_key,
    )
    tools ={}
    # Step 3: Create LangGraph Agent
    agent = create_react_agent(llm,tools)
    print(applicants)

    # Step 4: Build full prompt
    full_prompt = f"""
    {"Please provide constructive feedback for improvement based on the following applicant data." if not request.selected else "Congratulate the applicant for progressing to the next round."}
    Give only the feedback text
    Applicant Data:
    {applicants}
    """

    print("📤 Sending query to Gemini...",applicants)

    try:
        # Step 5: Ask LLM to translate query
        messages = [{"role": "user", "content": full_prompt}]
        response = await agent.ainvoke({"messages": messages})

        # Extract final text from agent
        final_output = None
        if isinstance(response, dict) and response.get("messages"):
            last_msg = response["messages"][-1]
            final_output = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else None)
        if not final_output:
            final_output = response.get("text") if isinstance(response, dict) else str(response)

        print("LLM output:\n", final_output)

    except Exception as e:
        error_msg = f"Failed to run agent: {str(e)}"
        print(f"❌ MongoDB Query failed: {error_msg}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": error_msg
        })

    result = send_student_mail(
        receiver_email=request.email,
        selected=request.selected,
        feedback=final_output or ""
    )

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return {"message": "Mail sent successfully to student."}

if __name__ == "__main__":
    
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)