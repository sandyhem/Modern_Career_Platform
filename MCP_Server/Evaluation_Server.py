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

# THE BELOW ARE THE MCP TOOLS FOR REPOSITORY PROFILES:

# async def fetch_repo_details(client, username, repo):
#     repo_name = repo["name"]
#     repo_url = repo["html_url"]

#     # Languages
#     languages_url = repo["languages_url"]
#     lang_resp = await client.get(languages_url)
#     languages = list(lang_resp.json().keys()) if lang_resp.status_code == 200 else []

#     # Basic stats
#     repo_data = {
#         "name": repo_name,
#         "url": repo_url,
#         "stars": repo.get("stargazers_count", 0),
#         "forks": repo.get("forks_count", 0),
#         "languages": languages,
#         "commits": 0
#     }

#     # Fetch total commits for default branch
#     default_branch = repo.get("default_branch")
#     if default_branch:
#         commits_url = f"{GITHUB_API}/repos/{username}/{repo_name}/commits?per_page=1&sha={default_branch}"
#         commits_resp = await client.get(commits_url)
#         if commits_resp.status_code == 200:
#             if "Link" in commits_resp.headers:
#                 import re
#                 match = re.search(r'&page=(\d+)>; rel="last"', commits_resp.headers["Link"])
#                 if match:
#                     repo_data["commits"] = int(match.group(1))
#             else:
#                 repo_data["commits"] = len(commits_resp.json())

#     return repo_data

# @mcp.tool(description="Get Github details of the user.")
# async def github_dashboard_analysis(username: str, token: str):
#     headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

#     async with httpx.AsyncClient(headers=headers, timeout=60) as client:
#         # Fetch user's repos
#         repos_resp = await client.get(f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner")
#         if repos_resp.status_code != 200:
#             return {"error": f"Failed to fetch repositories: {repos_resp.text}"}

#         repos = repos_resp.json()
#         if not repos:
#             return {"username": username, "message": "No repositories found"}

#         # Fetch repo details concurrently
#         tasks = [fetch_repo_details(client, username, repo) for repo in repos]
#         repo_details = await asyncio.gather(*tasks)

#         # Aggregate overall stats
#         total_repos = len(repo_details)
#         total_commits = sum(repo["commits"] for repo in repo_details)
#         total_stars = sum(repo["stars"] for repo in repo_details)
#         total_forks = sum(repo["forks"] for repo in repo_details)

#         all_languages_set = set()
#         for repo in repo_details:
#             all_languages_set.update(repo["languages"])
#         total_languages = len(all_languages_set)

#         return {
#             "username": username,
#             "total_repositories": total_repos,
#             "total_commits": total_commits,
#             "total_stars": total_stars,
#             "total_forks": total_forks,
#             "total_languages": total_languages,
#             "repos": repo_details
#         }

# @mcp.tool(description="Get GitHub Languages Proficiency used by a user")
# async def get_github_proficiency(username: str, token: str):
#     """
#     Analyze GitHub user's language usage and estimate proficiency.
#     """
#     headers = {"Accept": "application/vnd.github+json"}
#     if token:
#         headers["Authorization"] = f"token {token}"

#     async with httpx.AsyncClient(headers=headers, timeout=60) as client:
#         # Fetch repositories
#         repo_url = f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner&sort=updated"
#         repo_response = await client.get(repo_url)

#         if repo_response.status_code != 200:
#             return {"error": f"Error fetching repositories for {username}: {repo_response.text}"}

#         repos = repo_response.json()
#         if not repos:
#             return {"username": username, "message": "No public repositories found"}

#         all_languages = {}
#         repo_stats = {}
#         total_stars = 0
#         total_forks = 0
#         recent_commit_score = 0

#         for repo in repos:
#             repo_name = repo["name"]
#             stars = repo.get("stargazers_count", 0)
#             forks = repo.get("forks_count", 0)
#             total_stars += stars
#             total_forks += forks

#             # Last updated score (normalized)
#             updated_at = repo.get("updated_at")
#             if updated_at:
#                 delta_days = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at.replace("Z", "+00:00"))).days
#                 recent_commit_score += max(0, 100 - delta_days)  # more recent = higher score

#             # Fetch languages
#             languages_url = repo["languages_url"]
#             lang_response = await client.get(languages_url)
#             if lang_response.status_code == 200:
#                 langs = lang_response.json()
#                 for lang, bytes_used in langs.items():
#                     all_languages[lang] = all_languages.get(lang, 0) + bytes_used
#                     repo_stats[lang] = repo_stats.get(lang, 0) + 1

#         # Normalize data
#         total_bytes = sum(all_languages.values()) or 1
#         lang_percentage = {
#             lang: round((bytes_used / total_bytes) * 100, 2)
#             for lang, bytes_used in all_languages.items()
#         }

#         return {
#             "username": username,
#             "summary": {
#                 "total_repos": len(repos),
#                 "total_languages": len(all_languages),
#                 "total_stars": total_stars,
#                 "total_forks": total_forks,
#                 "recent_activity_score": recent_commit_score,
#             },
#             "languages": all_languages,
#             "language_percentage": lang_percentage,
#         }
    
# THE BELOW ARE THE MCP TOOLS FOR CODING PROFILES:

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

