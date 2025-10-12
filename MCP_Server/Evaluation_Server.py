import math
import httpx
import asyncio      
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from fastapi import HTTPException
import requests

GITHUB_API = "https://api.github.com"

mcp = FastMCP(name="Repository Server", stateless_http=True)

# THE BELOW ARE THE MCP TOOLS FOR REPOSITORY PROFILES:

async def fetch_repo_details(client, username, repo):
    repo_name = repo["name"]
    repo_url = repo["html_url"]

    # Languages
    languages_url = repo["languages_url"]
    lang_resp = await client.get(languages_url)
    languages = list(lang_resp.json().keys()) if lang_resp.status_code == 200 else []

    # Basic stats
    repo_data = {
        "name": repo_name,
        "url": repo_url,
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "languages": languages,
        "commits": 0
    }

    # Fetch total commits for default branch
    default_branch = repo.get("default_branch")
    if default_branch:
        commits_url = f"{GITHUB_API}/repos/{username}/{repo_name}/commits?per_page=1&sha={default_branch}"
        commits_resp = await client.get(commits_url)
        if commits_resp.status_code == 200:
            if "Link" in commits_resp.headers:
                import re
                match = re.search(r'&page=(\d+)>; rel="last"', commits_resp.headers["Link"])
                if match:
                    repo_data["commits"] = int(match.group(1))
            else:
                repo_data["commits"] = len(commits_resp.json())

    return repo_data

@mcp.tool(description="Get Github details of the user.")
async def github_dashboard_analysis(username: str, token: str):
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # Fetch user's repos
        repos_resp = await client.get(f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner")
        if repos_resp.status_code != 200:
            return {"error": f"Failed to fetch repositories: {repos_resp.text}"}

        repos = repos_resp.json()
        if not repos:
            return {"username": username, "message": "No repositories found"}

        # Fetch repo details concurrently
        tasks = [fetch_repo_details(client, username, repo) for repo in repos]
        repo_details = await asyncio.gather(*tasks)

        # Aggregate overall stats
        total_repos = len(repo_details)
        total_commits = sum(repo["commits"] for repo in repo_details)
        total_stars = sum(repo["stars"] for repo in repo_details)
        total_forks = sum(repo["forks"] for repo in repo_details)

        all_languages_set = set()
        for repo in repo_details:
            all_languages_set.update(repo["languages"])
        total_languages = len(all_languages_set)

        return {
            "username": username,
            "total_repositories": total_repos,
            "total_commits": total_commits,
            "total_stars": total_stars,
            "total_forks": total_forks,
            "total_languages": total_languages,
            "repos": repo_details
        }

@mcp.tool(description="Get GitHub Languages Proficiency used by a user")
async def get_github_proficiency(username: str, token: str):
    """
    Analyze GitHub user's language usage and estimate proficiency.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # Fetch repositories
        repo_url = f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner&sort=updated"
        repo_response = await client.get(repo_url)

        if repo_response.status_code != 200:
            return {"error": f"Error fetching repositories for {username}: {repo_response.text}"}

        repos = repo_response.json()
        if not repos:
            return {"username": username, "message": "No public repositories found"}

        all_languages = {}
        repo_stats = {}
        total_stars = 0
        total_forks = 0
        recent_commit_score = 0

        for repo in repos:
            repo_name = repo["name"]
            stars = repo.get("stargazers_count", 0)
            forks = repo.get("forks_count", 0)
            total_stars += stars
            total_forks += forks

            # Last updated score (normalized)
            updated_at = repo.get("updated_at")
            if updated_at:
                delta_days = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at.replace("Z", "+00:00"))).days
                recent_commit_score += max(0, 100 - delta_days)  # more recent = higher score

            # Fetch languages
            languages_url = repo["languages_url"]
            lang_response = await client.get(languages_url)
            if lang_response.status_code == 200:
                langs = lang_response.json()
                for lang, bytes_used in langs.items():
                    all_languages[lang] = all_languages.get(lang, 0) + bytes_used
                    repo_stats[lang] = repo_stats.get(lang, 0) + 1

        # Normalize data
        total_bytes = sum(all_languages.values()) or 1
        lang_percentage = {
            lang: round((bytes_used / total_bytes) * 100, 2)
            for lang, bytes_used in all_languages.items()
        }

        return {
            "username": username,
            "summary": {
                "total_repos": len(repos),
                "total_languages": len(all_languages),
                "total_stars": total_stars,
                "total_forks": total_forks,
                "recent_activity_score": recent_commit_score,
            },
            "languages": all_languages,
            "language_percentage": lang_percentage,
        }
    
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

# # Compute LeetCode score
# async def compute_leetcode_score(username: str):
#     url = f"https://leetcode-stats-api.herokuapp.com/{username}"
#     data = await safe_get(url)
#     print(data)
#     if not data or "totalSolved" not in data:
#         raise HTTPException(status_code=404, detail="Invalid LeetCode username")

#     total = data.get("totalSolved", 0)
#     ranking = data.get("ranking", 100000)
#     acceptance = data.get("acceptanceRate", 0)

#     problems_score = min(500, math.log10(total + 1) / math.log10(3000) * 500)
#     contest_score = (1 - min(1, ranking / 100000)) * 300
#     accuracy_score = min(1, acceptance / 100) * 200

#     return round(problems_score + contest_score + accuracy_score, 2), data

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

# # MCP tool: fetch LeetCode details and evaluate
# @mcp.tool(description="Fetch LeetCode user details and evaluate candidate")
# async def evaluate_leetcode_user(leetcode_username: str):
#     """
#     Fetches LeetCode stats for the given username and computes a final score with remarks.
#     Returns detailed stats and evaluation.
#     """
#     try:
#         score, data = await compute_leetcode_score(leetcode_username)
#     except HTTPException as e:
#         return {"error": str(e.detail)}

#     return {
#         "username": leetcode_username,
#         "final_score": score,
#         "remarks": remark(score),
#         "leetcode_details": data
#     }


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
    """
    Fetches LeetCode user stats including solved problems, contest ranking, and badges.
    """

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

    try:
        # ❗️requests is synchronous — use asyncio.to_thread to avoid blocking
        import asyncio
        response = await asyncio.to_thread(
            requests.post,
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (MCP-Agent)"
            },
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

    # Check HTTP status
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to fetch data from LeetCode"
        )

    data = response.json()

    # Validate response
    if "errors" in data or not data.get("data") or not data["data"].get("matchedUser"):
        raise HTTPException(
            status_code=404,
            detail="User not found or profile is private"
        )

    user_data = data["data"]["matchedUser"]
    contest_data = data["data"].get("userContestRanking", {})

    # 🧠 Extract problem stats
    ac_stats = {
        d["difficulty"]: d["count"]
        for d in user_data["submitStatsGlobal"]["acSubmissionNum"]
    }
    total_solved = sum(ac_stats.values())

    # 🐛 FIX: totalSubmissionNum is a list of dicts, so iterate properly
    total_submissions = 0
    for d in user_data["submitStatsGlobal"]["totalSubmissionNum"]:
        total_submissions += d["count"]

    # ✅ Compute acceptance rate safely
    acceptance_rate = round((total_solved / total_submissions) * 100, 2) if total_submissions > 0 else 0.0

    # 🧩 Combine results
    result = {
        "username": user_data["username"],
        "realName": user_data["profile"].get("realName"),
        "country": user_data["profile"].get("countryName"),
        "ranking": user_data["profile"].get("ranking"),
        "avatar": user_data["profile"].get("userAvatar"),
        "reputation": user_data["profile"].get("reputation"),
        "problemsSolved": {
            "Total": total_solved,
            "Easy": ac_stats.get("Easy", 0),
            "Medium": ac_stats.get("Medium", 0),
            "Hard": ac_stats.get("Hard", 0),
            "AcceptanceRate": f"{acceptance_rate}%",
        },
        "contestStats": {
            "AttendedContests": contest_data.get("attendedContestsCount"),
            "Rating": contest_data.get("rating"),
            "GlobalRanking": contest_data.get("globalRanking"),
            "TotalParticipants": contest_data.get("totalParticipants"),
            "TopPercentage": contest_data.get("topPercentage"),
        },
        "badges": [
            {
                "id": badge.get("id"),
                "name": badge.get("name"),
                "icon": badge.get("icon"),
                "earnedOn": badge.get("creationDate")
            }
            for badge in user_data.get("badges", [])
        ]
    }

    return result

