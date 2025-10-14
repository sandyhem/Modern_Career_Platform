"""
resume_matcher_backend.py
Backend API for Resume-based Job Matching with Gemini AI
Add these endpoints to your existing FastAPI application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
import PyPDF2
import docx
import io
import json
import re


from dotenv import load_dotenv
load_dotenv()

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Resume Matcher API",
        "gemini_configured": bool(os.getenv("GOOGLE_API_KEY"))
    }


# ==================== USAGE INSTRUCTIONS ====================
"""
SETUP INSTRUCTIONS:

1. Install required packages:
   pip install fastapi uvicorn PyPDF2 python-docx google-generativeai

2. Set environment variable:
   export GOOGLE_API_KEY="your-gemini-api-key"
   
   Windows:
   set GOOGLE_API_KEY=your-gemini-api-key

3. Run the server:
   uvicorn resume_matcher_backend:app --reload --port 8001

4. Test the endpoints:
   - Upload resume: POST http://localhost:8001/extract-resume-text
   - Analyze resume: POST http://localhost:8001/analyze-resume-gemini

FRONTEND INTEGRATION:

The frontend should:
1. Upload resume file to /extract-resume-text to get text
2. Send extracted text to /analyze-resume-gemini for AI analysis
3. Use returned keywords and skills to match with jobs in your database

EXAMPLE FLOW:
1. User uploads resume.pdf
2. POST to /extract-resume-text → get resume text
3. POST to /analyze-resume-gemini with text → get keywords, skills, experience
4. Frontend matches keywords with job descriptions
5. Show ranked job recommendations to user
"""

if __name__ == "__main__":
    import uvicorn
    print("🚀 Resume Matcher Backend starting on http://localhost:8001")
    print("📚 API Docs: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)