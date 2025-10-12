

# from fastapi import FastAPI, Query, HTTPException
# import requests, math

# app = FastAPI(title="Optimal Talent Scoring API")

# def safe_get(url: str):
#     try:
#         r = requests.get(url, timeout=10)
#         return r.json() if r.status_code == 200 else None
#     except:
#         return None

# def compute_leetcode_score(username: str):
#     url = f"https://leetcode-stats-api.herokuapp.com/{username}"
#     data = safe_get(url)
#     if not data or "totalSolved" not in data:
#         raise HTTPException(status_code=404, detail="Invalid LeetCode username")

#     total = data.get("totalSolved", 0)
#     ranking = data.get("ranking", 100000)
#     acceptance = data.get("acceptanceRate", 0)

#     problems_score = min(500, math.log10(total + 1) / math.log10(3000) * 500)
#     contest_score = (1 - min(1, ranking / 100000)) * 300
#     accuracy_score = min(1, acceptance / 100) * 200

#     return round(problems_score + contest_score + accuracy_score, 2), data

# def compute_codeforces_score(handle: str):
#     url = f"https://codeforces.com/api/user.info?handles={handle}"
#     data = safe_get(url)
#     if not data or data.get("status") != "OK":
#         return 0, None

#     user = data["result"][0]
#     rating = user.get("rating", 0)
#     contests = user.get("friendOfCount", 0)
#     contribution = user.get("contribution", 0)

#     base_score = min(1000, max(0, (rating - 800) / (3500 - 800) * 1000))
#     bonus = 0
#     if contests > 50: bonus += 50
#     if contribution > 0: bonus += 20

#     return round(min(1000, base_score + bonus), 2), user

# def remark(final_score):
#     if final_score < 300:
#         return "Beginner: still building fundamentals."
#     elif final_score < 600:
#         return "Intermediate: solid grasp of problem solving."
#     elif final_score < 800:
#         return "Advanced: strong contest-level programmer."
#     else:
#         return "Elite: top-tier algorithmic problem solver!"

# @app.get("/api/evaluate")
# def evaluate(leetcode_username: str, codeforces_username: str | None = None):
#     lc_score, lc_data = compute_leetcode_score(leetcode_username)
#     cf_score, cf_data = (0, None)
#     if codeforces_username:
#         cf_score, cf_data = compute_codeforces_score(codeforces_username)

#     final_score = max(lc_score, cf_score)
#     platform = "LeetCode" if lc_score >= cf_score else "Codeforces"

#     return {
#         "final_score": final_score,
#         "dominant_platform": platform,
#         "leetcode_score": lc_score,
#         "codeforces_score": cf_score,
#         "remarks": remark(final_score),
#         "leetcode_details": lc_data,
#         "codeforces_details": cf_data,
#     }


from fastapi import FastAPI, HTTPException
import requests

app = FastAPI(title="LeetCode Profile Stats API")

LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"

@app.get("/leetcode/{username}")
def get_full_leetcode_stats(username: str):
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
        response = requests.post(
            LEETCODE_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
            timeout=10
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from LeetCode")

    data = response.json()

    if "errors" in data or data.get("data", {}).get("matchedUser") is None:
        raise HTTPException(status_code=404, detail="User not found or profile is private")

    user_data = data["data"]["matchedUser"]
    contest_data = data["data"].get("userContestRanking", {})
    
    # Extract submission data
    ac_stats = {d["difficulty"]: d["count"] for d in user_data["submitStatsGlobal"]["acSubmissionNum"]}
    total_solved = sum(ac_stats.values())
    total_submissions = user_data["submitStatsGlobal"]["totalSubmissionNum"][0]["count"] if user_data["submitStatsGlobal"]["totalSubmissionNum"] else 0
    
    # Calculate acceptance rate
    acceptance_rate = round((total_solved / total_submissions) * 100, 2) if total_submissions > 0 else 0

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
                "id": badge["id"],
                "name": badge["name"],
                "icon": badge["icon"],
                "earnedOn": badge["creationDate"]
            }
            for badge in user_data.get("badges", [])
        ]
    }

    return result
