

from fastapi import FastAPI, Query, HTTPException
import requests, math

app = FastAPI(title="Optimal Talent Scoring API")

def safe_get(url: str):
    try:
        r = requests.get(url, timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def compute_leetcode_score(username: str):
    url = f"https://leetcode-stats-api.herokuapp.com/{username}"
    data = safe_get(url)
    if not data or "totalSolved" not in data:
        raise HTTPException(status_code=404, detail="Invalid LeetCode username")

    total = data.get("totalSolved", 0)
    ranking = data.get("ranking", 100000)
    acceptance = data.get("acceptanceRate", 0)

    problems_score = min(500, math.log10(total + 1) / math.log10(3000) * 500)
    contest_score = (1 - min(1, ranking / 100000)) * 300
    accuracy_score = min(1, acceptance / 100) * 200

    return round(problems_score + contest_score + accuracy_score, 2), data

def compute_codeforces_score(handle: str):
    url = f"https://codeforces.com/api/user.info?handles={handle}"
    data = safe_get(url)
    if not data or data.get("status") != "OK":
        return 0, None

    user = data["result"][0]
    rating = user.get("rating", 0)
    contests = user.get("friendOfCount", 0)
    contribution = user.get("contribution", 0)

    base_score = min(1000, max(0, (rating - 800) / (3500 - 800) * 1000))
    bonus = 0
    if contests > 50: bonus += 50
    if contribution > 0: bonus += 20

    return round(min(1000, base_score + bonus), 2), user

def remark(final_score):
    if final_score < 300:
        return "Beginner: still building fundamentals."
    elif final_score < 600:
        return "Intermediate: solid grasp of problem solving."
    elif final_score < 800:
        return "Advanced: strong contest-level programmer."
    else:
        return "Elite: top-tier algorithmic problem solver!"

@app.get("/api/evaluate")
def evaluate(leetcode_username: str, codeforces_username: str | None = None):
    lc_score, lc_data = compute_leetcode_score(leetcode_username)
    cf_score, cf_data = (0, None)
    if codeforces_username:
        cf_score, cf_data = compute_codeforces_score(codeforces_username)

    final_score = max(lc_score, cf_score)
    platform = "LeetCode" if lc_score >= cf_score else "Codeforces"

    return {
        "final_score": final_score,
        "dominant_platform": platform,
        "leetcode_score": lc_score,
        "codeforces_score": cf_score,
        "remarks": remark(final_score),
        "leetcode_details": lc_data,
        "codeforces_details": cf_data,
    }
