"""
mcp_server.py

Prototype MCP server (single-file) using FastAPI.

Features:
- Register students
- Add connectors/tokens per student (GitHub, Coursera)
- Aggregate achievements from connectors into a standard JSON format
- Simple recruiter API-key protection to fetch student aggregates

Requirements (install via pip):
    pip install fastapi "uvicorn[standard]" requests

Run:
    uvicorn mcp_server:app --reload --port 8000

OpenAPI docs after run:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Any
import uuid
import time
import requests  # used by connectors; if not available connectors fallback to mock

app = FastAPI(title="MCP - Student Achievements Aggregator", version="0.1")

#
# --- In-memory stores for demo ---
#
STUDENTS: Dict[str, Dict[str, Any]] = {}
# structure: STUDENTS[student_id] = {"name": ..., "connectors": {"github": {"token": "..."}}, "created": ...}

RECRUITER_KEYS: Dict[str, Dict[str, Any]] = {}
# RECRUITER_KEYS[api_key] = {"name": ..., "created": ...}


#
# --- Pydantic models ---
#
class StudentCreate(BaseModel):
    name: str = Field(..., example="Alice Student")
    email: Optional[str] = Field(None, example="alice@example.com")


class ConnectorAdd(BaseModel):
    provider: str = Field(..., example="github")
    token: str = Field(..., example="ghp_xxx")  # for demo; recommend OAuth in prod
    # optionally a username for public sources
    username: Optional[str] = Field(None, example="alice-gh")


class Achievement(BaseModel):
    student_id: str
    platform: str
    type: str  # e.g., "Project", "Certificate", "Badge", "Competition"
    title: str
    date: Optional[str] = None  # ISO date string
    url: Optional[HttpUrl] = None
    extra: Optional[Dict[str, Any]] = None


#
# --- Simple Recruiter auth dependency ---
#
def require_recruiter(api_key: str = Header(..., alias="X-API-Key")):
    if api_key not in RECRUITER_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing recruiter API key")
    return RECRUITER_KEYS[api_key]


#
# --- Connectors base and implementations ---
#
class ConnectorBase:
    provider_name: str

    def __init__(self, token: str, username: Optional[str] = None):
        self.token = token
        self.username = username

    def fetch_achievements(self) -> List[Dict[str, Any]]:
        """Return list of achievements in provider-specific raw form.
        Must be implemented by subclass.
        """
        raise NotImplementedError


class GithubConnector(ConnectorBase):
    provider_name = "github"

    def fetch_achievements(self) -> List[Dict[str, Any]]:
        """
        Attempt to fetch public repos for `username`. If token is provided and network available,
        fetch from GitHub API. Otherwise return mock data.
        """
        # If username provided, try to fetch; else fallback to token-based (not implemented)
        if not self.username:
            # cannot fetch without username in this simple demo -> mock
            return [
                {"type": "Project", "name": "mock-repo-1", "html_url": "https://github.com/mock/mock-repo-1", "created_at": "2024-05-01"},
                {"type": "Project", "name": "mock-repo-2", "html_url": "https://github.com/mock/mock-repo-2", "created_at": "2024-07-12"},
            ]

        url = f"https://api.github.com/users/{self.username}/repos"
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                repos = resp.json()
                out = []
                for r in repos:
                    out.append({
                        "type": "Project",
                        "name": r.get("name"),
                        "html_url": r.get("html_url"),
                        "created_at": r.get("created_at")
                    })
                return out
            else:
                # non-200 -> return a small mock with notice
                return [
                    {"type": "Project", "name": f"{self.username}-repo-sample", "html_url": f"https://github.com/{self.username}/repo-sample", "created_at": None}
                ]
        except Exception:
            # network or requests missing -> return mock
            return [
                {"type": "Project", "name": f"{self.username}-repo-sample", "html_url": f"https://github.com/{self.username}/repo-sample", "created_at": None}
            ]


class CourseraConnector(ConnectorBase):
    provider_name = "coursera"

    def fetch_achievements(self) -> List[Dict[str, Any]]:
        # Coursera requires OAuth and partner APIs; return mock demo entries
        # If you implement real integration, exchange OAuth token and call their records API
        return [
            {"type": "Certificate", "name": "Machine Learning - Coursera", "html_url": "https://coursera.org/verify/ABC123", "issued_date": "2023-11-12"},
            {"type": "Certificate", "name": "DeepLearning Specialization", "html_url": "https://coursera.org/verify/DEF456", "issued_date": "2024-02-20"},
        ]


# Map provider keyword -> connector class
CONNECTOR_MAP = {
    "github": GithubConnector,
    "coursera": CourseraConnector,
}


def instantiate_connector(provider: str, token: str, username: Optional[str] = None) -> ConnectorBase:
    cls = CONNECTOR_MAP.get(provider.lower())
    if not cls:
        raise HTTPException(status_code=400, detail=f"Provider `{provider}` is not supported.")
    return cls(token=token, username=username)


#
# --- Utility: normalize provider raw items to Achievement model ---
#
def normalize_to_achievement(student_id: str, provider: str, raw_item: Dict[str, Any]) -> Achievement:
    # simple mapping heuristics for demo connectors
    if provider == "github":
        return Achievement(
            student_id=student_id,
            platform="GitHub",
            type=raw_item.get("type", "Project"),
            title=raw_item.get("name") or raw_item.get("title"),
            date=raw_item.get("created_at"),
            url=raw_item.get("html_url"),
            extra={k: v for k, v in raw_item.items() if k not in ("name", "html_url", "created_at", "type")}
        )
    if provider == "coursera":
        return Achievement(
            student_id=student_id,
            platform="Coursera",
            type=raw_item.get("type", "Certificate"),
            title=raw_item.get("name"),
            date=raw_item.get("issued_date"),
            url=raw_item.get("html_url"),
            extra={k: v for k, v in raw_item.items() if k not in ("name", "html_url", "issued_date", "type")}
        )
    # fallback generic mapping
    return Achievement(
        student_id=student_id,
        platform=provider.title(),
        type=raw_item.get("type", "Achievement"),
        title=raw_item.get("name") or raw_item.get("title", "Untitled"),
        date=raw_item.get("date"),
        url=raw_item.get("url"),
        extra={k: v for k, v in raw_item.items() if k not in ("name", "title", "date", "url", "type")}
    )


#
# --- API endpoints ---
#
@app.post("/students", status_code=201)
def create_student(payload: StudentCreate):
    student_id = str(uuid.uuid4())
    STUDENTS[student_id] = {
        "name": payload.name,
        "email": payload.email,
        "connectors": {},
        "created": time.time()
    }
    return {"student_id": student_id, "message": "student registered"}


@app.post("/students/{student_id}/connectors", status_code=201)
def add_connector(student_id: str, payload: ConnectorAdd):
    student = STUDENTS.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    provider = payload.provider.lower()
    # store connector for demo; in prod store encrypted in DB
    student["connectors"][provider] = {
        "token": payload.token,
        "username": payload.username,
        "added": time.time()
    }
    return {"message": f"connector {provider} added to student {student_id}"}


@app.get("/students/{student_id}/achievements", response_model=List[Achievement])
def get_student_achievements(student_id: str):
    student = STUDENTS.get(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    achievements: List[Achievement] = []
    for prov, cfg in student["connectors"].items():
        try:
            conn = instantiate_connector(prov, token=cfg.get("token"), username=cfg.get("username"))
            raw_items = conn.fetch_achievements()
            for item in raw_items:
                achievements.append(normalize_to_achievement(student_id, prov, item))
        except HTTPException:
            # unsupported provider; skip
            continue
        except Exception:
            # connector failure -> skip but continue others
            continue
    # As a fallback, if no connectors or empty results, return empty list
    return achievements


@app.post("/recruiters/register", status_code=201)
def register_recruiter(name: str):
    api_key = str(uuid.uuid4())
    RECRUITER_KEYS[api_key] = {"name": name, "created": time.time()}
    return {"api_key": api_key, "message": "save this key securely; it is the recruiter's credential"}


@app.get("/recruiters/students/{student_id}/achievements", response_model=List[Achievement])
def recruiter_get_student_achievements(student_id: str, recruiter=Depends(require_recruiter)):
    # recruiter dependency ensures recruiter has provided valid X-API-Key header
    return get_student_achievements(student_id)


#
# Simple health
#
@app.get("/health")
def health():
    return {"status": "ok", "students_count": len(STUDENTS)}
