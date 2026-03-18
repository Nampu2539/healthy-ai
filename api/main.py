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

MODEL = "llama-3.3-70b-versatile"

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
    
    prompt = f"""คุณคือ Dr. AI ผู้เชี่ยวชาญด้านสุขภาพ วิเคราะห์ข้อมูลสุขภาพและให้คำแนะนำเฉพาะบุคคล

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
   - ประเภทการออกกำลังกายที่เหมาะสม
   - ความถี่และระยะเวลา

2. 🍎 **แผนโภชนาการ**
   - อาหารที่ควรเพิ่ม/ลด
   - เวลาและปริมาณที่เหมาะสม

3. 😴 **การนอนและสุขภาพจิต**
   - วิธีปรับปรุงคุณภาพการนอน
   - การจัดการความเครียด

4. ⚠️ **ความเสี่ยงที่ควรระวัง**
   - จุดที่ต้องให้ความสนใจเป็นพิเศษ

ตอบเป็นภาษาไทย กระชับ เข้าใจง่าย"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.5
    )
    
    return {"recommendation": response.choices[0].message.content}

@app.post("/symptom-check")
async def symptom_check(data: dict):
    symptoms = data.get("symptoms", "")
    age = data.get("age", "ไม่ระบุ")
    
    prompt = f"""ผู้ป่วยอายุ {age} ปี มีอาการ: {symptoms}

กรุณาวิเคราะห์อย่างละเอียดดังนี้:

1. 🔍 **การวิเคราะห์อาการ**
   - อธิบายอาการที่พบและความเชื่อมโยง

2. ⚠️ **ระดับความรุนแรง**
   - เบา / ปานกลาง / รุนแรง พร้อมเหตุผล

3. 💊 **การดูแลเบื้องต้น**
   - สิ่งที่ควรทำทันที
   - สิ่งที่ควรหลีกเลี่ยง

4. 🏥 **ควรพบแพทย์เมื่อไหร่**
   - สัญญาณเตือนที่ต้องรีบพบแพทย์

5. 🌿 **คำแนะนำเพิ่มเติม**
   - อาหาร การพักผ่อน การดูแลตัวเอง

⚠️ หมายเหตุ: นี่เป็นการวิเคราะห์เบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.3
    )
    
    return {"result": response.choices[0].message.content}

@app.post("/chat")
async def chat(data: dict):
    messages = data.get("messages", [])
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + messages,
        max_tokens=1024,
        temperature=0.7
    )
    
    return {"reply": response.choices[0].message.content}