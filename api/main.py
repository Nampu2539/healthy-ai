from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# โหลด CSV เดิมที่มีอยู่แล้ว
df = pd.read_csv("data/output/health_recommendations.csv")

@app.get("/")
def root():
    return {"status": "Health AI API Running ✅"}

@app.get("/user/{user_id}")
def get_user(user_id: int):
    user = df.iloc[user_id]
    return user.to_dict()

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    user = df.iloc[user_id]
    return {
        "exercise": user["Exercise_Recommendation"],
        "nutrition": user["Nutrition_Recommendation"],
        "lifestyle": user["Lifestyle_Recommendation"],
    }

@app.get("/analytics")
def get_analytics():
    return {
        "avg_wellness": float(df["Overall_Wellness_Score"].mean()),
        "avg_sleep": float(df["Sleep_Health_Score"].mean()),
        "total_users": len(df),
        "segments": df["User_Segment"].value_counts().to_dict()
    }