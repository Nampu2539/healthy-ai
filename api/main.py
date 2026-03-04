from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from groq import Groq
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

df = pd.read_csv("data/output/health_recommendations.csv")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
def root():
    return {"status": "Health AI API Running ✅"}

@app.get("/user/{user_id}")
def get_user(user_id: int):
    user = df.iloc[user_id]
    return user.to_dict()

@app.get("/analytics")
def get_analytics():
    return {
        "avg_wellness": float(df["Overall_Wellness_Score"].mean()),
        "avg_sleep": float(df["Sleep_Health_Score"].mean()),
        "total_users": len(df),
        "segments": df["User_Segment"].value_counts().to_dict()
    }

@app.post("/ai-recommend/{user_id}")
async def ai_recommend(user_id: int):
    user = df.iloc[user_id]
    
    prompt = f"""
    คุณเป็น AI Health Coach ผู้เชี่ยวชาญด้านสุขภาพ
    
    ข้อมูลสุขภาพของผู้ใช้:
    - อายุ: {user['Age']} ปี
    - BMI: {round(user['BMI'], 1)}
    - คะแนนการนอน: {user['Sleep_Health_Score']}/100
    - คะแนนการออกกำลังกาย: {user['Activity_Health_Score']}/100
    - คะแนนหัวใจ: {user['Cardiovascular_Health_Score']}/100
    - คะแนนสุขภาพจิต: {user['Mental_Health_Score']}/100
    - คะแนนรวม: {round(user['Overall_Wellness_Score'], 1)}/100
    
    กรุณาให้คำแนะนำที่:
    1. 🏋️ แผนออกกำลังกาย (2-3 ประโยค)
    2. 🍎 แผนโภชนาการ (2-3 ประโยค)
    3. 😴 คำแนะนำการนอนและจิตใจ (2-3 ประโยค)
    4. ⚠️ ความเสี่ยงที่ควรระวัง (1-2 ประโยค)
    
    ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    
    return {"recommendation": response.choices[0].message.content}
@app.post("/symptom-check")
async def symptom_check(data: dict):
    symptoms = data.get("symptoms", "")
    age = data.get("age", "ไม่ระบุ")
    
    prompt = f"""
    คุณเป็นแพทย์ AI ผู้เชี่ยวชาญ วิเคราะห์อาการเบื้องต้นเท่านั้น
    
    ข้อมูลผู้ป่วย:
    - อายุ: {age} ปี
    - อาการ: {symptoms}
    
    กรุณาวิเคราะห์:
    1. 🔍 การวิเคราะห์อาการเบื้องต้น
    2. ⚠️ ระดับความรุนแรง (เบา/ปานกลาง/รุนแรง)
    3. 💊 การดูแลตัวเองเบื้องต้น
    4. 🏥 ควรพบแพทย์เมื่อไหร่
    
    ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย
    ⚠️ แจ้งเตือนว่านี่เป็นเพียงการวิเคราะห์เบื้องต้น ไม่ใช่การวินิจฉัยทางการแพทย์
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    
    return {"result": response.choices[0].message.content}
@app.post("/chat")
async def chat(data: dict):
    messages = data.get("messages", [])
    
    system_prompt = {
        "role": "system",
        "content": """คุณเป็น AI Health Coach ผู้เชี่ยวชาญด้านสุขภาพ
        ให้คำแนะนำด้านการออกกำลังกาย โภชนาการ การนอนหลับ และสุขภาพจิต
        ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย ใช้ emoji ให้เหมาะสม
        เตือนให้พบแพทย์เมื่ออาการรุนแรง ไม่วินิจฉัยโรค"""
    }
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[system_prompt] + messages,
        max_tokens=1024
    )
    
    return {"reply": response.choices[0].message.content}