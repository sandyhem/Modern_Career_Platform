"""
github_mcp_server.py
A simple MCP-style Python server that aggregates all GitHub details of a user.
Author: ChatGPT (GPT-5)

Run:
    pip install fastapi "uvicorn[standard]" requests
    uvicorn github_mcp_server:app --reload --port 8000
"""
# Set environment variable before running: $env:GEMINI_API_KEY="AIzaSyDJQJe-qibVUD0aUhp-X5NEI83W5Lryyg0"

import requests
import re  # Add this import
import time
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(
    title="GitHub MCP Server",
    description="Fetches all GitHub data (profile, repos, orgs, gists, events) for a user",
    version="1.0.0"
)

GITHUB_API_BASE = "https://api.github.com"


# --------- Models ---------
class GitHubUser(BaseModel):
    login: str
    name: str | None = None
    bio: str | None = None
    public_repos: int
    followers: int
    following: int
    created_at: str
    updated_at: str
    avatar_url: str
    html_url: str


class Repo(BaseModel):
    name: str
    html_url: str
    description: str | None = None
    stargazers_count: int
    forks_count: int
    language: str | None = None
    updated_at: str


class Organization(BaseModel):
    login: str
    description: str | None = None
    url: str
    avatar_url: str


class Gist(BaseModel):
    id: str
    html_url: str
    description: str | None = None
    created_at: str


class GitHubUserData(BaseModel):
    profile: GitHubUser
    repositories: list[Repo]
    organizations: list[Organization]
    gists: list[Gist]
    events: list[dict]


# Add this missing model
class ScoreResponse(BaseModel):
    username: str
    activity_score: float
    relevance_score: float
    final_score: float
    breakdown: dict


# --------- Helper Functions ---------
def github_api_request(endpoint: str, token: str | None = None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    resp = requests.get(f"{GITHUB_API_BASE}{endpoint}", headers=headers, timeout=10)
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="User or resource not found.")
    elif resp.status_code == 403:
        raise HTTPException(status_code=403, detail="API rate limit exceeded or access denied.")
    elif not resp.ok:
        raise HTTPException(status_code=resp.status_code, detail=f"GitHub API error: {resp.text}")
    return resp.json()

def get_readme_content(username: str, repo_name: str, token: str | None = None):
    """Fetch README content from a repository with rate limiting."""
    try:
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        
        # Try common README filenames
        readme_files = ['README.md', 'readme.md', 'README.txt', 'README.rst', 'README']
        
        for readme_file in readme_files:
            try:
                headers = {"Accept": "application/vnd.github+json"}
                if token:
                    headers["Authorization"] = f"token {token}"
                
                url = f"{GITHUB_API_BASE}/repos/{username}/{repo_name}/contents/{readme_file}"
                resp = requests.get(url, headers=headers, timeout=10)
                
                # Handle rate limiting
                if resp.status_code == 403:
                    print(f"Rate limit hit for {repo_name}, skipping README")
                    return ""
                
                if resp.status_code == 200:
                    content_info = resp.json()
                    if 'download_url' in content_info and content_info['download_url']:
                        readme_resp = requests.get(content_info['download_url'], timeout=5)
                        if readme_resp.status_code == 200:
                            return readme_resp.text.lower()
                    break
            except Exception as e:
                print(f"Error fetching README for {repo_name}: {str(e)}")
                continue
        return ""
    except Exception:
        return ""

# --------- Endpoints ---------
@app.get("/user/{username}", response_model=GitHubUserData)
def get_user_data(username: str, token: str | None = Query(None)):
    profile = github_api_request(f"/users/{username}", token)
    repos = github_api_request(f"/users/{username}/repos?per_page=100", token)

    return GitHubUserData(
        profile=GitHubUser(**profile),
        repositories=[Repo(**{
            "name": r["name"],
            "html_url": r["html_url"],
            "description": r.get("description"),
            "stargazers_count": r.get("stargazers_count", 0),
            "forks_count": r.get("forks_count", 0),
            "language": r.get("language"),
            "updated_at": r["updated_at"],
        }) for r in repos]
    )


@app.post("/score/{username}", response_model=ScoreResponse)
def compute_score(username: str, jd: str = Query(..., description="Job description text"), token: str | None = Query(None)):
    profile = github_api_request(f"/users/{username}", token)
    repos = github_api_request(f"/users/{username}/repos?per_page=100", token)

    # ---------- ENHANCED JD KEYWORD EXTRACTION ----------
    jd_lower = jd.lower()
    
    # Extract specific technology keywords with weights
    tech_keywords = {
        'python': 3, 'javascript': 3, 'java': 3, 'react': 3, 'node': 3,
        'angular': 3, 'vue': 3, 'typescript': 3, 'go': 2, 'rust': 2,
        'docker': 2, 'kubernetes': 2, 'aws': 2, 'azure': 2, 'gcp': 2,
        'sql': 2, 'mysql': 2, 'postgresql': 2, 'mongodb': 2, 'redis': 2,
        'flask': 2, 'django': 2, 'express': 2, 'fastapi': 2, 'spring': 2,
        'tensorflow': 2, 'pytorch': 2, 'pandas': 2, 'numpy': 2, 'opencv': 2,
        'git': 1, 'linux': 1, 'bash': 1, 'ci': 1, 'cd': 1, 'jenkins': 1
    }
    
    # Find keywords present in JD with their weights
    jd_tech_found = {}
    for tech, weight in tech_keywords.items():
        if tech in jd_lower:
            jd_tech_found[tech] = weight
    
    # Extract general keywords from JD
    jd_keywords = set(re.findall(r"\b[a-zA-Z0-9\+\#]+\b", jd_lower))
    
    # ---------- ACTIVITY METRICS ----------
    commits = sum([r.get("stargazers_count", 0) + r.get("forks_count", 0) for r in repos])
    stars = sum([r.get("stargazers_count", 0) for r in repos])
    forks = sum([r.get("forks_count", 0) for r in repos])
    followers = profile.get("followers", 0)
    total_repos = len(repos)

    activity_score = 0
    activity_score += min(commits / 8, 18)   # More generous
    activity_score += min(stars / 4, 12)      
    activity_score += min(forks / 2, 10)       
    activity_score += min(followers / 3, 10)   
    activity_score += min(total_repos / 1.5, 15) # More weight on repo count
    
    if total_repos > 0:
        activity_score += 15  # Higher base bonus

    activity_score = min(activity_score, 70)  # Higher cap

    # ---------- ENHANCED RELEVANCE METRICS ----------
    weighted_matches = 0
    lang_matches = 0
    topic_matches = 0
    ci_cd = 0
    testing = 0
    readme_quality = 0
    specific_tech_score = 0

    for r in repos:
        repo_text = f"{r.get('name', '')} {r.get('description', '')}".lower()
        
        # Weighted technology matching
        for tech, weight in jd_tech_found.items():
            if tech in repo_text or (r.get("language") and tech in r["language"].lower()):
                specific_tech_score += weight
        
        # Language matching with JD context
        if r.get("language"):
            lang = r["language"].lower()
            if lang in jd_keywords:
                lang_matches += 2  # Higher weight for exact matches
            elif any(kw in lang for kw in jd_keywords):
                lang_matches += 1
        
        # Enhanced description matching
        desc = (r.get("description") or "").lower()
        repo_name = (r.get("name") or "").lower()
        
        # Count JD keyword matches with context
        jd_matches_in_repo = sum(1 for kw in jd_keywords if kw in desc or kw in repo_name)
        topic_matches += min(jd_matches_in_repo, 3)  # Cap per repo but accumulate
        
        # Quality indicators
        if r.get("description") and len(r.get("description", "")) > 15:
            readme_quality += 1
        if r.get("stargazers_count", 0) > 0:
            readme_quality += 1  # Stars indicate quality
            
        # CI/CD with more indicators
        ci_cd_indicators = ['docker', 'ci', 'cd', 'pipeline', 'deploy', 'build', 'action', 'github', 'workflow', 'jenkins', 'travis']
        ci_cd_score = sum(1 for indicator in ci_cd_indicators if indicator in desc or indicator in repo_name)
        ci_cd += min(ci_cd_score, 2)  # Max 2 points per repo
            
        # Testing with more indicators
        test_indicators = ['test', 'spec', 'jest', 'pytest', 'unittest', 'mocha', 'testing', 'tdd', 'junit', 'karma']
        test_score = sum(1 for indicator in test_indicators if indicator in desc or indicator in repo_name)
        testing += min(test_score, 2)  # Max 2 points per repo

    # ---------- DYNAMIC RELEVANCE SCORING ----------
    relevance_score = 0
    
    # Technology-specific bonus (higher if JD mentions specific tech)
    relevance_score += min(specific_tech_score * 2, 25)
    
    # Language matching (scaled by JD requirements)
    relevance_score += min(lang_matches * 2, 20)
    
    # Topic matching (scaled by JD keyword density)
    jd_keyword_density = len(jd_tech_found) / max(len(jd_keywords), 1)
    topic_multiplier = 1 + jd_keyword_density  # 1x to 2x multiplier
    relevance_score += min(topic_matches * topic_multiplier, 25)
    
    # DevOps/Quality practices
    relevance_score += min(ci_cd * 1.5, 12)
    relevance_score += min(testing * 1.5, 12)
    relevance_score += min(readme_quality * 1, 8)
    
    # Dynamic base score based on JD complexity
    if total_repos > 0:
        base_relevance = 15 + (len(jd_tech_found) * 2)  # Higher base for complex JDs
        relevance_score += min(base_relevance, 25)

    relevance_score = min(relevance_score, 80)  # Higher cap for relevance

    # ---------- FINAL SCORE WITH JD ADAPTATION ----------
    # Weight based on JD technical complexity
    jd_complexity = len(jd_tech_found) / 10  # 0 to 1+ scale
    activity_weight = max(0.2, 0.4 - jd_complexity)  # Less weight on activity for complex JDs
    relevance_weight = 1 - activity_weight
    
    final_score = round((activity_score * activity_weight) + (relevance_score * relevance_weight), 2)
    
    # Bonus system based on JD requirements
    if specific_tech_score > 5:  # Strong tech match
        final_score += 8
    elif specific_tech_score > 2:  # Some tech match
        final_score += 4
        
    if total_repos >= 5 and len(jd_tech_found) > 3:  # Active developer for complex role
        final_score += 6
    elif total_repos >= 3:
        final_score += 3
        
    if followers >= 5:  # Community recognition
        final_score += 4
    elif followers >= 2:
        final_score += 2
        
    final_score = min(final_score, 100)

    return ScoreResponse(
        username=username,
        activity_score=round(activity_score, 2),
        relevance_score=round(relevance_score, 2),
        final_score=final_score,
        breakdown={
            "total_repos": total_repos,
            "commits_proxy": commits,
            "stars": stars,
            "forks": forks,
            "followers": followers,
            "lang_matches": lang_matches,
            "topic_matches": topic_matches,
            "specific_tech_matches": specific_tech_score,
            "jd_tech_found": list(jd_tech_found.keys()),
            "jd_complexity_score": len(jd_tech_found),
            "readme_quality": readme_quality,
            "ci_cd": ci_cd,
            "testing": testing,
        }
    )

# --------- API Route for Repo Contents ---------
@app.get("/github/{username}/{repo}/contents")
def get_repo_contents_endpoint(username: str, repo: str):
    """Fetch the contents of a GitHub repository (files/folders in root)."""
    url = f"https://api.github.com/repos/{username}/{repo}/contents"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail=f"GitHub API error: {response.text}")


# --------- API Route ---------
@app.get("/github/{username}", response_model=GitHubUserData)
def get_github_user(
    username: str,
    token: str | None = Query(None, description="Optional GitHub PAT for private or authenticated access")
):
    """Fetch full GitHub details for the given username."""

    # --- 1. Basic user info ---
    user_json = github_api_request(f"/users/{username}", token)
    print(user_json)
    profile = GitHubUser(**user_json)

    # --- 2. Repositories ---
    repos_json = github_api_request(f"/users/{username}/repos?per_page=100&sort=updated", token)
    repos = [
        Repo(
            name=r["name"],
            html_url=r["html_url"],
            description=r.get("description"),
            stargazers_count=r["stargazers_count"],
            forks_count=r["forks_count"],
            language=r.get("language"),
            updated_at=r["updated_at"]
        ) for r in repos_json
    ]

    # --- 3. Organizations ---
    # Use authenticated user's orgs endpoint if token is provided to get both public and private memberships
    if token:
        try:
            # Try to get authenticated user's organizations (includes private memberships)
            auth_user_json = github_api_request("/user", token)
            if auth_user_json["login"].lower() == username.lower():
                # If the token belongs to the requested user, get all their orgs (public + private)
                orgs_json = github_api_request("/user/orgs", token)
            else:
                # If token belongs to different user, fall back to public orgs only
                orgs_json = github_api_request(f"/users/{username}/orgs", token)
        except:
            # If there's any error with authenticated endpoints, fall back to public orgs
            orgs_json = github_api_request(f"/users/{username}/orgs", token)
    else:
        # No token provided, can only get public organizations
        orgs_json = github_api_request(f"/users/{username}/orgs", token)
    
    orgs = [
        Organization(
            login=o["login"],
            description=o.get("description"),
            url=o["url"],
            avatar_url=o["avatar_url"]
        ) for o in orgs_json
    ]

    # --- 4. Gists ---
    gists_json = github_api_request(f"/users/{username}/gists", token)
    gists = [
        Gist(
            id=g["id"],
            html_url=g["html_url"],
            description=g.get("description"),
            created_at=g["created_at"]
        ) for g in gists_json
    ]

    # --- 5. Events (recent activity) ---
    events_json = github_api_request(f"/users/{username}/events/public", token)

    return GitHubUserData(
        profile=profile,
        repositories=repos,
        organizations=orgs,
        gists=gists,
        events=events_json[:10]  # limit to last 10 events
    )




@app.get("/github/{username}/private-repos")
def get_private_repos(
    username: str,
    token: str = Query(..., description="GitHub PAT required for accessing private repositories")
):
    """
    Fetch only private repositories for the given username.
    Requires a valid GitHub Personal Access Token with repo scope.
    """
    if not token:
        raise HTTPException(
            status_code=400, 
            detail="GitHub token is required to access private repositories"
        )
    
    # Use authenticated user endpoint if username matches token owner, 
    # otherwise use user-specific endpoint
    try:
        # First, try to get user's own repos (this will include private ones if token belongs to user)
        repos_json = github_api_request(f"/user/repos?per_page=100&sort=updated&type=private", token)
        
        # Filter to only include repos owned by the specified username
        filtered_repos = []
        for repo in repos_json:
            if repo["owner"]["login"].lower() == username.lower() and repo["private"]:
                filtered_repos.append(repo)
        
        # If no repos found with user endpoint, try the public user endpoint 
        # (this won't show private repos but will verify user exists)
        if not filtered_repos:
            # Verify user exists
            github_api_request(f"/users/{username}", token)
            # Return empty list if user exists but no private repos accessible
            return []
        
        # Convert to Repo models
        private_repos = [
            Repo(
                name=r["name"],
                html_url=r["html_url"],
                description=r.get("description"),
                stargazers_count=r["stargazers_count"],
                forks_count=r["forks_count"],
                language=r.get("language"),
                updated_at=r["updated_at"]
            ) for r in filtered_repos
        ]
        
        return {
            "username": username,
            "private_repositories_count": len(private_repos),
            "private_repositories": private_repos
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching private repositories: {str(e)}")


@app.get("/github/{username}/metrics")
def get_github_metrics(
    username: str,
    token: str | None = Query(None, description="Optional GitHub PAT for authenticated access")
):
    """
    Aggregates useful GitHub metrics for recruiters:
    - Total public contributions (commits)
    - Pull requests raised
    - Issues opened
    - Stars received
    - Forks made
    """
    # Fetch repositories
    repos_json = github_api_request(f"/users/{username}/repos?per_page=100", token)
    total_commits = 0
    total_stars = 0
    total_forks = 0

    for repo in repos_json:
        try:
            # Commits: fetch commit count for each repo
            commits_url = f"/repos/{username}/{repo['name']}/commits?per_page=1"
            headers = {"Accept": "application/vnd.github+json"}
            if token:
                headers["Authorization"] = f"token {token}"
            
            commits_resp = requests.get(f"{GITHUB_API_BASE}{commits_url}", headers=headers, timeout=10)
            
            if commits_resp.status_code == 200:
                if "Link" in commits_resp.headers:
                    # Parse last page from Link header for total commits
                    import re
                    match = re.search(r'&page=(\d+)>; rel="last"', commits_resp.headers["Link"])
                    if match:
                        total_commits += int(match.group(1))
                    else:
                        total_commits += len(commits_resp.json()) if commits_resp.json() else 0
                else:
                    commits_data = commits_resp.json()
                    total_commits += len(commits_data) if commits_data else 0
        except Exception:
            # Skip this repo if commit fetching fails
            continue
            
        total_stars += repo.get("stargazers_count", 0)
        total_forks += repo.get("forks_count", 0)

    # Pull requests raised
    try:
        pulls_json = github_api_request(f"/search/issues?q=author:{username}+type:pr", token)
        total_pull_requests = pulls_json.get("total_count", 0)
    except Exception:
        total_pull_requests = 0

    try:
        # Issues opened
        issues_json = github_api_request(f"/search/issues?q=author:{username}+type:issue", token)
        total_issues = issues_json.get("total_count", 0)
    except Exception:
        total_issues = 0

    return {
        "username": username,
        "total_public_repos": len(repos_json),
        "total_commits": total_commits,
        "total_pull_requests": total_pull_requests,
        "total_issues_opened": total_issues,
        "total_stars_received": total_stars,
        "total_forks_made": total_forks
    }


@app.post("/score-simple/{username}", response_model=ScoreResponse)
def compute_score_simple(username: str, jd: str = Query(..., description="Job description text"), token: str | None = Query(None)):
    """Simplified scoring without README fetching to avoid rate limits."""
    profile = github_api_request(f"/users/{username}", token)
    repos = github_api_request(f"/users/{username}/repos?per_page=100", token)

    # ---------- ACTIVITY METRICS (More Liberal) ----------
    commits = sum([r.get("stargazers_count", 0) + r.get("forks_count", 0) for r in repos])  # proxy
    stars = sum([r.get("stargazers_count", 0) for r in repos])
    forks = sum([r.get("forks_count", 0) for r in repos])
    followers = profile.get("followers", 0)
    total_repos = len(repos)

    activity_score = 0
    # More generous activity scoring
    activity_score += min(commits / 10, 15)   
    activity_score += min(stars / 5, 10)      
    activity_score += min(forks / 3, 8)       
    activity_score += min(followers / 5, 7)   
    activity_score += min(total_repos / 2, 10) 
    
    # Base score for having any GitHub activity
    if total_repos > 0:
        activity_score += 10

    if activity_score > 60:
        activity_score = 60

    # ---------- RELEVANCE METRICS (More Liberal, No README) ----------
    jd_keywords = set(re.findall(r"\b[a-zA-Z0-9\+\#]+\b", jd.lower()))
    
    # Add common tech keywords
    common_keywords = {'python', 'javascript', 'java', 'react', 'node', 'sql', 'html', 'css', 'git'}
    jd_keywords.update(common_keywords)

    lang_matches = 0
    topic_matches = 0
    ci_cd = 0
    testing = 0

    for r in repos:
        # Language matching
        if r.get("language"):
            lang = r["language"].lower()
            if lang in jd_keywords or any(kw in lang for kw in jd_keywords):
                lang_matches += 1
        
        # Description matching
        desc = (r.get("description") or "").lower()
        repo_name = (r.get("name") or "").lower()
        
        if any(kw in desc or kw in repo_name for kw in jd_keywords):
            topic_matches += 1
        
        # CI/CD indicators
        ci_cd_indicators = ['docker', 'ci', 'cd', 'pipeline', 'deploy', 'build', 'action']
        if any(indicator in desc or indicator in repo_name for indicator in ci_cd_indicators):
            ci_cd += 1
            
        # Testing indicators
        test_indicators = ['test', 'spec', 'jest', 'pytest', 'unittest', 'mocha']
        if any(indicator in desc or indicator in repo_name for indicator in test_indicators):
            testing += 1

    relevance_score = 0
    relevance_score += min(lang_matches * 3, 20)      
    relevance_score += min(topic_matches * 3, 20)     # increased since no README
    relevance_score += min(ci_cd * 3, 10)             
    relevance_score += min(testing * 3, 10)           
    
    # Base relevance score
    if total_repos > 0:
        relevance_score += 20  # higher base since no README analysis

    if relevance_score > 65:
        relevance_score = 65

    # ---------- FINAL SCORE ----------
    final_score = round((activity_score * 0.3) + (relevance_score * 0.7), 2)
    
    # Minimum score boost for active developers
    if total_repos >= 3:
        final_score += 5
    if followers >= 2:
        final_score += 3
        
    # Cap at 100
    final_score = min(final_score, 100)

    return ScoreResponse(
        username=username,
        activity_score=round(activity_score, 2),
        relevance_score=round(relevance_score, 2),
        final_score=final_score,
        breakdown={
            "total_repos": total_repos,
            "commits_proxy": commits,
            "stars": stars,
            "forks": forks,
            "followers": followers,
            "lang_matches": lang_matches,
            "topic_matches": topic_matches,
            "readme_matches": 0,  # Not fetched in simple version
            "readme_quality": 0,  # Not fetched in simple version
            "ci_cd": ci_cd,
            "testing": testing,
        }
    )


@app.post("/gemini-keywords/{username}")
def gemini_keyword_match(username: str, jd: str = Body(..., embed=True), token: str | None = Query(None)):
    """
    Uses Gemini AI to extract and match keywords between job description and candidate profile.
    Returns matched keywords, missing skills, and compatibility score.
    """
    try:
        profile = github_api_request(f"/users/{username}", token)
        repos = github_api_request(f"/users/{username}/repos?per_page=50", token)

        # Build candidate profile text
        languages = list(set([r.get('language') for r in repos if r.get('language')]))
        repo_descriptions = [r.get('description', '') for r in repos if r.get('description')]
        
        candidate_profile = f"""
        GitHub Profile: {username}
        Bio: {profile.get('bio', 'No bio available')}
        Programming Languages: {', '.join(languages)}
        Repository Descriptions: {' | '.join(repo_descriptions[:10])}
        Repository Names: {', '.join([r['name'] for r in repos[:20]])}
        """

        # Create keyword extraction prompt
        prompt = f"""
        Extract and match technical keywords between this job description and candidate profile.

        JOB DESCRIPTION:
        {jd}

        CANDIDATE PROFILE:
        {candidate_profile}

        Please provide a JSON-style response with:

        1. JD_KEYWORDS: List all technical skills, technologies, frameworks, and tools mentioned in the job description
        2. CANDIDATE_KEYWORDS: List all technical skills, technologies, frameworks, and tools found in the candidate profile
        3. MATCHED_KEYWORDS: List keywords that appear in both JD and candidate profile
        4. MISSING_KEYWORDS: List important keywords from JD that are missing from candidate profile
        5. COMPATIBILITY_SCORE: Give a percentage (0-100) based on keyword overlap
        6. KEYWORD_CATEGORIES: Group keywords by category (languages, frameworks, tools, databases, etc.)

        Focus only on technical skills and technologies. Ignore soft skills or general terms.
        Be precise with technology names (e.g., distinguish React from React Native, Python from JavaScript).
        """

        # Call Gemini API
        if not os.getenv("GEMINI_API_KEY"):
            return {
                "username": username,
                "error": "Gemini API key not configured",
                "message": "Set GEMINI_API_KEY environment variable"
            }

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return {
            "username": username,
            "job_description_preview": jd[:200] + "..." if len(jd) > 200 else jd,
            "candidate_summary": {
                "languages": languages,
                "total_repos": len(repos),
                "bio": profile.get('bio', 'No bio')[:100] if profile.get('bio') else 'No bio'
            },
            "gemini_keyword_analysis": response.text,
            "extraction_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {
            "username": username,
            "error": f"Failed to extract keywords: {str(e)}",
            "suggestion": "Check if username exists and Gemini API key is properly set"
        }


@app.get("/health")
def health_check():
    return {"status": "ok"}
