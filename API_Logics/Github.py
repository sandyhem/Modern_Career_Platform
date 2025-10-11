from fastapi import FastAPI, HTTPException, Query
import httpx
from datetime import datetime, timezone
import math
import asyncio

app = FastAPI(title="GitHub Language Fetcher with Token")

GITHUB_API = "https://api.github.com"

# for language fetcher

@app.get("/github/{username}/languages")
async def get_github_languages(username: str, token: str = Query(None, description="GitHub Personal Access Token")):
    """
    Fetch languages used in all public repositories of a GitHub user.
    Optional: Provide a GitHub PAT token for higher rate limits.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers) as client:
        # Step 1: Fetch repositories
        repo_url = f"https://api.github.com/users/{username}/repos?per_page=100"
        repo_response = await client.get(repo_url)

        if repo_response.status_code != 200:
            raise HTTPException(
                status_code=repo_response.status_code,
                detail=f"Error fetching repositories for {username}. GitHub says: {repo_response.text}",
            )

        repos = repo_response.json()


        if not repos:
            return {"username": username, "languages": {}, "message": "No public repositories found"}

        all_languages = {}

        # Step 2: Fetch languages from each repo
        for repo in repos:
            languages_url = repo["languages_url"]
            lang_response = await client.get(languages_url)

            if lang_response.status_code == 200:
                langs = lang_response.json()
                for lang, bytes_used in langs.items():
                    all_languages[lang] = all_languages.get(lang, 0) + bytes_used

        # Step 3: Sort and format
        sorted_languages = dict(
            sorted(all_languages.items(), key=lambda x: x[1], reverse=True)
        )

        # Step 4: Compute total and percentage
        total_bytes = sum(sorted_languages.values())
        percentage = {
            lang: round((bytes_used / total_bytes) * 100, 2)
            for lang, bytes_used in sorted_languages.items()
        }

        return {
            "username": username,
            "total_languages": len(sorted_languages),
            "languages": sorted_languages,
            "language_percentage": percentage,
        }

# sample trials

@app.get("/github/{username}/language-proficiency")
async def get_github_proficiency(
    username: str,
    token: str = Query(None, description="GitHub Personal Access Token for authentication"),
):
    """
    Analyze GitHub user's language usage and estimate proficiency.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers) as client:
        # Fetch repositories
        repo_url = f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner&sort=updated"
        repo_response = await client.get(repo_url)

        if repo_response.status_code != 200:
            raise HTTPException(
                status_code=repo_response.status_code,
                detail=f"Error fetching repositories for {username}: {repo_response.text}",
            )

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
                    repo_stats.setdefault(lang, 0)
                    repo_stats[lang] += 1

        # Normalize data
        total_bytes = sum(all_languages.values())
        lang_percentage = {
            lang: round((bytes_used / total_bytes) * 100, 2)
            for lang, bytes_used in all_languages.items()
        }

        # # Compute a simple heuristic proficiency score
        # language_proficiency = {}
        # for lang, usage_percent in lang_percentage.items():
        #     repos_using = repo_stats.get(lang, 1)
        #     # Weighted score = code usage + repo count + activity + popularity
        #     score = (
        #         (usage_percent * 0.5)
        #         + (repos_using * 5)
        #         + (total_stars * 0.01)
        #         + (recent_commit_score * 0.02)
        #     )
        #     language_proficiency[lang] = round(score, 2)

        # sorted_proficiency = dict(
        #     sorted(language_proficiency.items(), key=lambda x: x[1], reverse=True)
        # )

        return {
            "username": username,
            "summary": {
                "total_repos": len(repos),
                "total_languages": len(all_languages),
                "total_stars": total_stars,
                "total_forks": total_forks,
            },
            "languages": all_languages,
            "language_percentage": lang_percentage,
            # "estimated_proficiency": sorted_proficiency,
        }

@app.get("/github/{username}/github-stats")
async def get_github_stats(
    username: str,
    token: str = Query(None, description="GitHub Personal Access Token for authentication"),
):
    """
    Analyze GitHub user's language usage and estimate proficiency.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers) as client:
        # Fetch repositories
        repo_url = f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner&sort=updated"
        repo_response = await client.get(repo_url)

        if repo_response.status_code != 200:
            raise HTTPException(
                status_code=repo_response.status_code,
                detail=f"Error fetching repositories for {username}: {repo_response.text}",
            )

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
                    repo_stats.setdefault(lang, 0)
                    repo_stats[lang] += 1

        # Normalize data
        total_bytes = sum(all_languages.values())
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
            },
            "languages": all_languages,
            "language_percentage": lang_percentage,
            # "estimated_proficiency": sorted_proficiency,
        }

# for skill score analysis

@app.get("/github/{username}/skill-score")
async def get_github_skill_score(
    username: str,
    token: str = Query(None, description="GitHub Personal Access Token for authentication"),
):
    """
    Analyze a GitHub user's language proficiency using:
    Skill Score = 0.4 * code_volume + 0.3 * repo_diversity + 0.2 * complexity + 0.1 * documentation
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(headers=headers) as client:
        # Step 1: Get user's repos
        repo_url = f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner"
        repo_response = await client.get(repo_url)

        if repo_response.status_code != 200:
            raise HTTPException(
                status_code=repo_response.status_code,
                detail=f"Error fetching repositories for {username}: {repo_response.text}",
            )

        repos = repo_response.json()
        if not repos:
            return {"username": username, "message": "No public repositories found"}

        all_languages = {}
        repo_languages = {}
        repo_complexity = {}  # repo → total language bytes
        repo_docs = {}  # repo → has README or docs folder

        # Step 2: Collect language + documentation info
        for repo in repos:
            repo_name = repo["name"]
            languages_url = repo["languages_url"]
            contents_url = f"{GITHUB_API}/repos/{username}/{repo_name}/contents"

            # Fetch languages used
            lang_resp = await client.get(languages_url)
            if lang_resp.status_code == 200:
                langs = lang_resp.json()
                repo_total = sum(langs.values())
                repo_complexity[repo_name] = repo_total
                for lang, bytes_used in langs.items():
                    all_languages[lang] = all_languages.get(lang, 0) + bytes_used
                    repo_languages.setdefault(lang, set()).add(repo_name)

            # Fetch contents to check documentation
            docs_resp = await client.get(contents_url)
            has_docs = False
            if docs_resp.status_code == 200:
                contents = docs_resp.json()
                for item in contents:
                    if item["type"] == "file" and "readme" in item["name"].lower():
                        has_docs = True
                        break
                    if item["type"] == "dir" and item["name"].lower() in ["docs", "documentation"]:
                        has_docs = True
                        break
            repo_docs[repo_name] = has_docs

        # Step 3: Compute weights per language
        total_code_volume = sum(all_languages.values())
        if total_code_volume == 0:
            raise HTTPException(status_code=400, detail="No measurable code volume found.")

        skill_scores = {}
        for lang, bytes_used in all_languages.items():
            code_volume_weight = bytes_used / total_code_volume  # normalized
            repo_diversity_weight = len(repo_languages.get(lang, [])) / len(repos)

            # Average complexity across repos using this language
            lang_repos = repo_languages.get(lang, [])
            if lang_repos:
                avg_complexity = sum(repo_complexity[r] for r in lang_repos) / len(lang_repos)
            else:
                avg_complexity = 0
            complexity_weight = math.log1p(avg_complexity) / 20  # normalize using log scale

            # Documentation presence score
            docs_present = sum(1 for r in lang_repos if repo_docs.get(r))
            documentation_weight = docs_present / len(lang_repos) if lang_repos else 0

            # Final skill score
            score = (
                (0.4 * code_volume_weight)
                + (0.3 * repo_diversity_weight)
                + (0.2 * complexity_weight)
                + (0.1 * documentation_weight)
            )

            skill_scores[lang] = round(score * 100, 2)  # Scale to 0–100

        sorted_skills = dict(
            sorted(skill_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "username": username,
            "total_repositories": len(repos),
            "total_languages": len(all_languages),
            "languages": all_languages,
            "skill_scores": sorted_skills,
            "interpretation": "Higher score indicates stronger proficiency inferred from code volume, diversity, complexity, and documentation presence."
        }

#  for dashboard analysis

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
        "commits": 0  # will calculate next
    }

    # Fetch total commits for default branch
    default_branch = repo.get("default_branch")
    if default_branch:
        commits_url = f"{GITHUB_API}/repos/{username}/{repo_name}/commits?per_page=1&sha={default_branch}"
        commits_resp = await client.get(commits_url)
        if commits_resp.status_code == 200:
            # GitHub provides total commits in Link header (last page)
            if "Link" in commits_resp.headers:
                import re
                match = re.search(r'&page=(\d+)>; rel="last"', commits_resp.headers["Link"])
                if match:
                    repo_data["commits"] = int(match.group(1))
            else:
                repo_data["commits"] = len(commits_resp.json())

    return repo_data

@app.get("/github/{username}/dashboard-analysis")
async def github_dashboard_analysis(
    username: str,
    token: str = Query(..., description="GitHub Personal Access Token")
):
    """
    Returns aggregated GitHub statistics for a candidate for recruiter dashboard.
    Includes: total repos, commits per repo, stars, forks, languages, repo URLs.
    """
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        # Fetch user's repos
        repos_resp = await client.get(f"{GITHUB_API}/users/{username}/repos?per_page=100&type=owner")
        if repos_resp.status_code != 200:
            raise HTTPException(status_code=repos_resp.status_code, detail="Failed to fetch repositories")

        repos = repos_resp.json()
        if not repos:
            return {"username": username, "message": "No repositories found"}

        # Fetch each repo details concurrently
        tasks = [fetch_repo_details(client, username, repo) for repo in repos]
        repo_details = await asyncio.gather(*tasks)

        # Aggregate overall stats
        total_repos = len(repo_details)
        total_commits = sum(repo["commits"] for repo in repo_details)
        total_stars = sum(repo["stars"] for repo in repo_details)
        total_forks = sum(repo["forks"] for repo in repo_details)

        # Collect all languages across repos
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
