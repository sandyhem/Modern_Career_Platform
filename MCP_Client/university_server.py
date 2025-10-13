"""
university_server.py
Backend server for University Student Verification Portal
Handles Excel upload and stores data in MySQL database
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import pandas as pd
import io
import os

# ==================== DATABASE SETUP ====================
DATABASE_URL = os.getenv(
    "MYSQL_URL",
    "mysql+pymysql://root:root@localhost:3306/universityDB"
)


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== DATABASE MODELS ====================
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
    internships = Column(Text, nullable=True)  # Stored as JSON string
    verification_status = Column(String(20), default="Pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC MODELS ====================
class StudentVerification(BaseModel):
    register_number: str = Field(..., description="University register number")
    name: str = Field(..., description="Full name of the student")
    department: str = Field(..., description="Department of study")
    year_of_study: int = Field(..., ge=1, le=4, description="Current year of study (1 to 4)")
    cgpa: float = Field(..., ge=0.0, le=10.0, description="Cumulative Grade Point Average")
    total_backlogs: int = Field(0, ge=0, description="Total number of past backlogs cleared")
    current_arrears: int = Field(0, ge=0, description="Number of arrears currently uncleared")
    history_of_arrears: bool = Field(False, description="True if the student had arrears anytime")
    placement_eligible: bool = Field(False, description="Whether the student meets eligibility criteria for placement")
    internships: Optional[List[str]] = Field(default_factory=list, description="Internships completed by the student")
    verification_status: str = Field("Pending", description="Status of verification - Pending / Verified / Rejected")

class StudentVerificationResponse(StudentVerification):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    success: bool
    message: str
    records_added: int
    records_updated: int
    errors: List[str] = []

class UpdateStatusRequest(BaseModel):
    verification_status: str = Field(..., description="New status: Pending / Verified / Rejected")

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="University Student Verification Portal",
    description="API for managing student verification data",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATABASE DEPENDENCY ====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "message": "University Student Verification Portal API",
        "endpoints": {
            "upload": "/upload-excel",
            "students": "/students",
            "student_by_regno": "/students/{register_number}",
            "update_status": "/students/{register_number}/status",
            "stats": "/stats"
        }
    }

@app.post("/upload-excel", response_model=UploadResponse)
async def upload_excel(file: UploadFile = File(...)):
    """
    Upload Excel file with student verification data.
    Expected columns: register_number, name, department, year_of_study, cgpa, 
                     total_backlogs, current_arrears, history_of_arrears, 
                     placement_eligible, internships, verification_status
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")
    
    try:
        # Read Excel file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['register_number', 'name', 'department', 'year_of_study', 'cgpa']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        db = next(get_db())
        records_added = 0
        records_updated = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Parse internships (comma-separated string to list)
                internships_str = str(row.get('internships', ''))
                internships_list = [i.strip() for i in internships_str.split(',') if i.strip()]
                
                # Check if student already exists
                existing = db.query(StudentVerificationDB).filter(
                    StudentVerificationDB.register_number == str(row['register_number'])
                ).first()
                
                if existing:
                    # Update existing record
                    existing.name = str(row['name'])
                    existing.department = str(row['department'])
                    existing.year_of_study = int(row['year_of_study'])
                    existing.cgpa = float(row['cgpa'])
                    existing.total_backlogs = int(row.get('total_backlogs', 0))
                    existing.current_arrears = int(row.get('current_arrears', 0))
                    existing.history_of_arrears = bool(row.get('history_of_arrears', False))
                    existing.placement_eligible = bool(row.get('placement_eligible', False))
                    existing.internships = ','.join(internships_list) if internships_list else None
                    existing.verification_status = str(row.get('verification_status', 'Pending'))
                    existing.updated_at = datetime.utcnow()
                    records_updated += 1
                else:
                    # Create new record
                    new_student = StudentVerificationDB(
                        register_number=str(row['register_number']),
                        name=str(row['name']),
                        department=str(row['department']),
                        year_of_study=int(row['year_of_study']),
                        cgpa=float(row['cgpa']),
                        total_backlogs=int(row.get('total_backlogs', 0)),
                        current_arrears=int(row.get('current_arrears', 0)),
                        history_of_arrears=bool(row.get('history_of_arrears', False)),
                        placement_eligible=bool(row.get('placement_eligible', False)),
                        internships=','.join(internships_list) if internships_list else None,
                        verification_status=str(row.get('verification_status', 'Pending'))
                    )
                    db.add(new_student)
                    records_added += 1
                
            except Exception as e:
                errors.append(f"Row {index + 2}: {str(e)}")
        
        db.commit()
        
        return UploadResponse(
            success=True,
            message=f"Excel processed successfully",
            records_added=records_added,
            records_updated=records_updated,
            errors=errors
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

@app.get("/students", response_model=List[StudentVerificationResponse])
def get_all_students(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    department: Optional[str] = None,
    verification_status: Optional[str] = None,
    placement_eligible: Optional[bool] = None
):
    """Get all students with optional filters"""
    db = next(get_db())
    
    query = db.query(StudentVerificationDB)
    
    if department:
        query = query.filter(StudentVerificationDB.department == department)
    if verification_status:
        query = query.filter(StudentVerificationDB.verification_status == verification_status)
    if placement_eligible is not None:
        query = query.filter(StudentVerificationDB.placement_eligible == placement_eligible)
    
    students = query.offset(skip).limit(limit).all()
    
    # Convert internships string back to list
    result = []
    for student in students:
        student_dict = {
            "id": student.id,
            "register_number": student.register_number,
            "name": student.name,
            "department": student.department,
            "year_of_study": student.year_of_study,
            "cgpa": student.cgpa,
            "total_backlogs": student.total_backlogs,
            "current_arrears": student.current_arrears,
            "history_of_arrears": student.history_of_arrears,
            "placement_eligible": student.placement_eligible,
            "internships": student.internships.split(',') if student.internships else [],
            "verification_status": student.verification_status,
            "created_at": student.created_at,
            "updated_at": student.updated_at
        }
        result.append(StudentVerificationResponse(**student_dict))
    
    return result

@app.get("/students/{register_number}", response_model=StudentVerificationResponse)
def get_student_by_regno(register_number: str):
    """Get student by registration number"""
    db = next(get_db())
    
    student = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.register_number == register_number
    ).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    student_dict = {
        "id": student.id,
        "register_number": student.register_number,
        "name": student.name,
        "department": student.department,
        "year_of_study": student.year_of_study,
        "cgpa": student.cgpa,
        "total_backlogs": student.total_backlogs,
        "current_arrears": student.current_arrears,
        "history_of_arrears": student.history_of_arrears,
        "placement_eligible": student.placement_eligible,
        "internships": student.internships.split(',') if student.internships else [],
        "verification_status": student.verification_status,
        "created_at": student.created_at,
        "updated_at": student.updated_at
    }
    
    return StudentVerificationResponse(**student_dict)

@app.put("/students/{register_number}/status")
def update_verification_status(register_number: str, status_update: UpdateStatusRequest):
    """Update verification status of a student"""
    db = next(get_db())
    
    student = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.register_number == register_number
    ).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    valid_statuses = ["Pending", "Verified", "Rejected"]
    if status_update.verification_status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    student.verification_status = status_update.verification_status
    student.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": f"Status updated to {status_update.verification_status}", "register_number": register_number}

@app.delete("/students/{register_number}")
def delete_student(register_number: str):
    """Delete a student record"""
    db = next(get_db())
    
    student = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.register_number == register_number
    ).first()
    
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.delete(student)
    db.commit()
    
    return {"message": "Student record deleted successfully", "register_number": register_number}

@app.get("/stats")
def get_statistics():
    """Get overall statistics"""
    db = next(get_db())
    
    total_students = db.query(StudentVerificationDB).count()
    verified = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.verification_status == "Verified"
    ).count()
    pending = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.verification_status == "Pending"
    ).count()
    rejected = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.verification_status == "Rejected"
    ).count()
    placement_eligible = db.query(StudentVerificationDB).filter(
        StudentVerificationDB.placement_eligible == True
    ).count()
    
    # Department-wise count
    departments = db.query(
        StudentVerificationDB.department,
        db.func.count(StudentVerificationDB.id)
    ).group_by(StudentVerificationDB.department).all()
    
    return {
        "total_students": total_students,
        "verified": verified,
        "pending": pending,
        "rejected": rejected,
        "placement_eligible": placement_eligible,
        "departments": {dept: count for dept, count in departments}
    }

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "University Portal API"}


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    print("🎓 University Portal Server starting on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)