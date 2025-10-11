"""
coding_profiles_mcp_server.py
MCP-style API server that aggregates data from LeetCode, GFG, and Codeforces.
Author: ChatGPT (GPT-5)

Run:
    uvicorn coding_profiles_mcp_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

app = FastAPI(
    title="Coding Profile MCP Server",
    description="Fetches LeetCode, GFG, and Codeforces profile stats",
    version="1.0.0"
)

# ---------- MODELS ----------
class LeetCodeProfile(BaseModel):
    username: str
    ranking: int | None = None
    total_problems_solved: int | None = None
    easy_solved: int | None = None
    medium_solved: int | None = None
    hard_solved: int | None = None
    contest_rating: float | None = None
    profile_url: str


class GFGProfile(BaseModel):
    username: str
    total_problems_solved: int | None = None
    coding_score: int | None = None
    institute_rank: int | None = None
    profile_url: str


class CodeforcesProfile(BaseModel):
    username: str
    rating: int | None = None
    max_rating: int | None = None
    rank: str | None = None
    contribution: int | None = None
    profile_url: str


class CombinedProfile(BaseModel):
    leetcode: LeetCodeProfile | None
    gfg: GFGProfile | None
    codeforces: CodeforcesProfile | None


# ---------- FETCH FUNCTIONS ----------

def fetch_leetcode_profile(username: str):
    """Fetch LeetCode profile data via unofficial API."""
    url = f"https://leetcode-stats-api.herokuapp.com/{username}"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="LeetCode profile not found")

    data = resp.json()
    return LeetCodeProfile(
        username=username,
        ranking=data.get("ranking"),
        total_problems_solved=data.get("totalSolved"),
        easy_solved=data.get("easySolved"),
        medium_solved=data.get("mediumSolved"),
        hard_solved=data.get("hardSolved"),
        contest_rating=data.get("contestRating"),
        profile_url=f"https://leetcode.com/{username}/"
    )


def fetch_gfg_profile(username: str):
    """Fetch GeeksforGeeks profile data using HTML scraping."""
    url = f"https://auth.geeksforgeeks.org/user/{username}/practice/"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="GFG profile not found")

    soup = BeautifulSoup(resp.text, "html.parser")
    stats = {"total_problems_solved": None, "coding_score": None, "institute_rank": None}

    # Parse problem count and score
    try:
        stats_tags = soup.find_all("span", class_="score_card_value")
        if len(stats_tags) >= 2:
            stats["coding_score"] = int(stats_tags[0].text.strip())
            stats["total_problems_solved"] = int(stats_tags[1].text.strip())
    except Exception:
        pass

    # Parse rank (if available)
    try:
        rank_tag = soup.find("div", text=lambda x: x and "Institution Rank" in x)
        if rank_tag:
            stats["institute_rank"] = int(rank_tag.find_next("div").text.strip())
    except Exception:
        pass

    return GFGProfile(
        username=username,
        total_problems_solved=stats["total_problems_solved"],
        coding_score=stats["coding_score"],
        institute_rank=stats["institute_rank"],
        profile_url=url
    )


def fetch_codeforces_profile(username: str):
    """Fetch Codeforces profile using official public API."""
    url = f"https://codeforces.com/api/user.info?handles={username}"
    resp = requests.get(url, timeout=10)

    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Codeforces profile not found")

    data = resp.json()
    if data["status"] != "OK":
        raise HTTPException(status_code=404, detail="Invalid Codeforces username")

    user = data["result"][0]
    return CodeforcesProfile(
        username=user["handle"],
        rating=user.get("rating"),
        max_rating=user.get("maxRating"),
        rank=user.get("rank"),
        contribution=user.get("contribution"),
        profile_url=f"https://codeforces.com/profile/{username}"
    )


# ---------- ROUTE ----------
@app.get("/profiles", response_model=CombinedProfile)
def get_profiles(
    leetcode: str | None = Query(None),
    gfg: str | None = Query(None),
    codeforces: str | None = Query(None)
):
    """Fetches user profile data from multiple platforms."""
    result = CombinedProfile(leetcode=None, gfg=None, codeforces=None)

    if leetcode:
        result.leetcode = fetch_leetcode_profile(leetcode)

    if gfg:
        result.gfg = fetch_gfg_profile(gfg)

    if codeforces:
        result.codeforces = fetch_codeforces_profile(codeforces)

    if not any([leetcode, gfg, codeforces]):
        raise HTTPException(status_code=400, detail="At least one username must be provided.")

    return result


@app.get("/health")
def health_check():
    return {"status": "ok"}
