from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import google.generativeai as genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

df = pd.read_csv("data/output/health_recommendations.csv")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

SYSTEM_PROMPT = """คุณคือ Dr. AI ผู้เชี่ยวชาญด้านสุขภาพและการแพทย์ที่มีประสบการณ์กว่า 20 ปี

ความเชี่ยวชาญ:
- โภชนาการและการควบคุมอาหาร
- การออกกำลังกายและฟื้นฟูร่างกาย
- สุขภาพจิตและการจัดการความเครียด
- โรคเรื้อรังและการป้องกัน
- การนอนหลับและการฟื้นตัว

วิธีตอบ:
- ตอบเป็นภาษาไทยเสมอ
- ใช้ภาษาที่เข้าใจง่าย ไม่ใช้ศัพท์แพทย์มากเกินไป
- ให้คำแนะนำที่เป็นรูปธรรม ทำได้จริง
- แสดงความห่วงใยและเข้าใจผู้ป่วย
- ใช้ emoji ประกอบให้เหมาะสม
- ถ้าอาการรุนแรง ให้แนะนำพบแพทย์ทันที
- ไม่วินิจฉัยโรคโดยตรง แต่ให้ข้อมูลเบื้องต้นได้"""

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
    
    prompt = f"""{SYSTEM_PROMPT}

ข้อมูลสุขภาพของผู้ใช้:
- อายุ: {user['Age']} ปี
- BMI: {round(user['BMI'], 1)}
- คะแนนการนอน: {user['Sleep_Health_Score']}/100
- คะแนนการออกกำลังกาย: {user['Activity_Health_Score']}/100
- คะแนนหัวใจ: {user['Cardiovascular_Health_Score']}/100
- คะแนนสุขภาพจิต: {user['Mental_Health_Score']}/100
- คะแนนรวม: {round(user['Overall_Wellness_Score'], 1)}/100

กรุณาให้คำแนะนำที่ละเอียดและเป็นรูปธรรม:

1. 🏋️ **แผนออกกำลังกาย**
2. 🍎 **แผนโภชนาการ**
3. 😴 **การนอนและสุขภาพจิต**
4. ⚠️ **ความเสี่ยงที่ควรระวัง**"""

    response = model.generate_content(prompt)
    return {"recommendation": response.text}

@app.post("/symptom-check")
async def symptom_check(data: dict):
    symptoms = data.get("symptoms", "")
    age = data.get("age", "ไม่ระบุ")
    
    prompt = f"""{SYSTEM_PROMPT}

ผู้ป่วยอายุ {age} ปี มีอาการ: {symptoms}

กรุณาวิเคราะห์อย่างละเอียด:

1. 🔍 **การวิเคราะห์อาการ**
2. ⚠️ **ระดับความรุนแรง** (เบา/ปานกลาง/รุนแรง)
3. 💊 **การดูแลเบื้องต้น**
4. 🏥 **ควรพบแพทย์เมื่อไหร่**
5. 🌿 **คำแนะนำเพิ่มเติม**

⚠️ นี่เป็นการวิเคราะห์เบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์"""

    response = model.generate_content(prompt)
    return {"result": response.text}

@app.post("/chat")
async def chat(data: dict):
    messages = data.get("messages", [])
    
    chat_session = model.start_chat(history=[
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [m["content"]]
        }
        for m in messages[:-1]
    ])
    
    last_message = messages[-1]["content"] if messages else ""
    full_prompt = f"{SYSTEM_PROMPT}\n\n{last_message}"
    
    response = chat_session.send_message(full_prompt)
    return {"reply": response.text}