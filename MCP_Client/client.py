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
students_collection = db["students"]
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

if __name__ == "__main__":
    
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000,reload=True)