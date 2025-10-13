import math
import httpx,re
import asyncio      
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException
import requests
import os

from dotenv import load_dotenv
load_dotenv()

GITHUB_API = "https://api.github.com"

mcp = FastMCP(name="Repository Server", stateless_http=True)

# Async GET request
async def safe_get(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json()
            return None
    except:
        return None

# Generate remarks
def remark(final_score: float):
    if final_score < 300:
        return "Beginner: still building fundamentals."
    elif final_score < 600:
        return "Intermediate: solid grasp of problem solving."
    elif final_score < 800:
        return "Advanced: strong contest-level programmer."
    else:
        return "Elite: top-tier algorithmic problem solver!"

# Compute Codeforces score
async def compute_codeforces_score(handle: str):
    url = f"https://codeforces.com/api/user.info?handles={handle}"
    data = await safe_get(url)
    if not data or data.get("status") != "OK":
        raise HTTPException(status_code=404, detail="Invalid Codeforces handle")

    user = data["result"][0]
    rating = user.get("rating", 0)
    contests = user.get("friendOfCount", 0)
    contribution = user.get("contribution", 0)

    # Base score based on rating
    base_score = min(1000, max(0, (rating - 800) / (3500 - 800) * 1000))
    bonus = 0
    if contests > 50:
        bonus += 50
    if contribution > 0:
        bonus += 20

    final_score = round(min(1000, base_score + bonus), 2)
    return final_score, user

# MCP tool: fetch Codeforces details and evaluate
@mcp.tool(description="Fetch Codeforces user details and evaluate candidate")
async def evaluate_codeforces_user(codeforces_handle: str):
    """
    Fetches Codeforces stats for the given handle and computes a final score with remarks.
    Returns detailed stats and evaluation.
    """
    try:
        score, data = await compute_codeforces_score(codeforces_handle)
    except HTTPException as e:
        return {"error": str(e.detail)}

    return {
        "handle": codeforces_handle,
        "final_score": score,
        "remarks": remark(score),
        "codeforces_details": data
    }

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"


@mcp.tool(description="Fetch LeetCode user details and evaluate candidate")
async def evaluate_leetcode_user(username: str):
    query = """
    query getUserStats($username: String!) {
      matchedUser(username: $username) {
        username
        profile {
          realName
          ranking
          reputation
          userAvatar
          countryName
        }
        submitStatsGlobal {
          acSubmissionNum {
            difficulty
            count
            submissions
          }
          totalSubmissionNum {
            count
          }
        }
        badges {
          id
          name
          icon
          creationDate
        }
      }
      userContestRanking(username: $username) {
        attendedContestsCount
        rating
        globalRanking
        totalParticipants
        topPercentage
      }
    }
    """
    variables = {"username": username}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.post(
                LEETCODE_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (MCP-Agent)"}
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from LeetCode")

    data = response.json()
    matched_user = data.get("data", {}).get("matchedUser")
    if not matched_user:
        raise HTTPException(status_code=404, detail="User not found or profile is private")

    profile = matched_user.get("profile") or {}
    real_name = profile.get("realName")
    country = profile.get("countryName")
    ranking = profile.get("ranking")
    avatar = profile.get("userAvatar")
    reputation = profile.get("reputation")

    submit_stats = matched_user.get("submitStatsGlobal") or {}
    ac_submissions = submit_stats.get("acSubmissionNum") or []
    total_submissions_list = submit_stats.get("totalSubmissionNum") or []

    ac_stats = {d.get("difficulty"): d.get("count", 0) for d in ac_submissions if d}
    total_solved = sum(ac_stats.values())
    total_submissions = sum(d.get("count", 0) for d in total_submissions_list)
    acceptance_rate = round((total_solved / total_submissions) * 100, 2) if total_submissions > 0 else 0.0

    contest_data = data.get("data", {}).get("userContestRanking") or {}
    contest_stats = {
        "AttendedContests": contest_data.get("attendedContestsCount", 0),
        "Rating": contest_data.get("rating", 0),
        "GlobalRanking": contest_data.get("globalRanking", 0),
        "TotalParticipants": contest_data.get("totalParticipants", 0),
        "TopPercentage": contest_data.get("topPercentage", 0.0),
    }

    badges = [
        {"id": badge.get("id"), "name": badge.get("name"), "icon": badge.get("icon"), "earnedOn": badge.get("creationDate")}
        for badge in (matched_user.get("badges") or [])
    ]

    return {
        "username": matched_user.get("username"),
        "realName": real_name,
        "country": country,
        "ranking": ranking,
        "avatar": avatar,
        "reputation": reputation,
        "problemsSolved": {
            "Total": total_solved,
            "Easy": ac_stats.get("Easy", 0),
            "Medium": ac_stats.get("Medium", 0),
            "Hard": ac_stats.get("Hard", 0),
            "AcceptanceRate": f"{acceptance_rate}%",
        },
        "contestStats": contest_stats,
        "badges": badges
    }

@mcp.tool(description="Analyze GitHub candidate profile against job description and compute compatibility score")
async def analyze_candidate(username: str, job_description: str):
    """
    Analyze a GitHub user's repositories, activity, and languages,
    and compute a compatibility score against a given job description.
    """
    token = os.getenv("PAT_TOKEN")
    
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # --- Fetch profile ---
        profile_res = await client.get(f"{GITHUB_API}/users/{username}")
        if profile_res.status_code != 200:
            return {"error": f"Failed to fetch profile: {profile_res.text}"}
        profile = profile_res.json()

        # --- Fetch repositories ---
        repo_res = await client.get(f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner&sort=updated")
        if repo_res.status_code != 200:
            return {"error": f"Failed to fetch repos: {repo_res.text}"}
        repos = repo_res.json()
        if not repos:
            return {"username": username, "message": "No public repositories found"}

        # --- GitHub stats ---
        total_repos = len(repos)
        total_stars = sum(r.get("stargazers_count", 0) for r in repos)
        total_forks = sum(r.get("forks_count", 0) for r in repos)
        total_watchers = sum(r.get("watchers_count", 0) for r in repos)
        followers = profile.get("followers", 0)

        # --- Calculate total commits (approximation via pagination) ---
        total_commits = 0
        for repo in repos:
            repo_name = repo.get("name")
            owner = repo.get("owner", {}).get("login")
            commits_url = f"{GITHUB_API}/repos/{owner}/{repo_name}/commits"
            commits_res = await client.get(commits_url, params={"per_page": 1})
            if commits_res.status_code != 200:
                continue
            link_header = commits_res.headers.get("Link")
            if link_header and 'rel="last"' in link_header:
                try:
                    last_page = int(link_header.split("page=")[-1].split(">")[0])
                    total_commits += last_page
                except Exception:
                    total_commits += 1
            else:
                total_commits += len(commits_res.json())

        # --- Aggregate language usage ---
        lang_totals = {}
        for repo in repos:
            lang_url = repo.get("languages_url")
            lang_res = await client.get(lang_url)
            if lang_res.status_code != 200:
                continue
            for lang, bytes_used in lang_res.json().items():
                lang_totals[lang] = lang_totals.get(lang, 0) + bytes_used

        total_bytes = sum(lang_totals.values()) or 1
        language_weights = {k: v / total_bytes for k, v in lang_totals.items()}

        # --- Match job description ---
        jd_keywords = re.findall(r"\b[a-zA-Z0-9\+\#]+\b", job_description.lower())
        matched_langs = [lang for lang in language_weights if any(k in lang.lower() for k in jd_keywords)]

        # calculate weighted match score
        match_score = sum(language_weights.get(lang, 0) for lang in matched_langs)
        if matched_langs:
            match_score += 0.4  # boost if at least one match found
        match_score = min(0.85, match_score)

        # --- Normalize total commits ---
        # cap at 1 for >=10k commits
        commit_factor = min(1, total_commits / 150)

        # --- Compatibility score (70% language match, 30% commits) ---
        compatibility_score = round(((0.7 * match_score) + (0.3 * commit_factor)) * 100, 2)

        # --- Return ---
        return {
            "username": username,
            "job_description_preview": job_description,
            "languages": lang_totals,
            "matched_keywords": matched_langs,
            "github_activity": {
                "total_repos": total_repos,
                "total_stars": total_stars,
                "total_forks": total_forks,
                "total_watchers": total_watchers,
                "followers": followers,
                "total_commits": total_commits,
            },
            "commit_factor":commit_factor,
            "match_score":match_score,
            "compatibility_score": compatibility_score
        }

