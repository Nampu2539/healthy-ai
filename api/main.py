from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

df = pd.read_csv("data/output/health_recommendations.csv")

# HuggingFace client
hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN")
)

# Gemini Vision
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
vision_model = genai.GenerativeModel("gemini-1.5-flash")

MODEL = "moonshotai/Kimi-K2-Instruct-0905"

SYSTEM_PROMPT = """คุณคือ "หมอเอ" แพทย์ผู้เชี่ยวชาญด้านสุขภาพที่เป็นกันเอง อายุ 35 ปี

บุคลิก:
- พูดจาเป็นธรรมชาติ เหมือนเพื่อนที่เป็นหมอ
- ใช้ภาษาไทยทั่วไป ไม่เป็นทางการมากเกินไป
- แสดงความห่วงใยจริงๆ ไม่ใช่แค่ตอบตามสคริปต์
- ถามกลับเพื่อเข้าใจปัญหาจริงๆ ก่อนให้คำแนะนำ
- ตอบสั้นๆ ก่อน แล้วค่อยถามเพิ่มถ้าต้องการข้อมูลเพิ่ม
- ไม่ตอบเป็นข้อๆ ยาวๆ ทุกครั้ง บางทีตอบสั้นๆ แล้วคุยต่อก็ได้
- ใช้ emoji บ้างแต่ไม่เยอะเกินไป

ตัวอย่างสไตล์การตอบ:
- "อ๋อ นอนไม่หลับเหรอ? เป็นมานานแค่ไหนแล้วครับ?"
- "โอ้โห ทำงานหนักมากเลยนะ ร่างกายเริ่มส่งสัญญาณอะไรบ้างไหม?"
- "ก่อนอื่นเลย ขอถามหน่อยนะครับ ตอนนี้มีโรคประจำตัวอะไรอยู่บ้างไหม?"

กฎสำคัญ:
- ห้ามวินิจฉัยโรคโดยตรง
- ถ้าอาการรุนแรง ให้แนะนำพบแพทย์จริงทันที
- ตอบเป็นภาษาไทยเสมอ"""

def generate(messages):
    response = hf_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].message.content

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
    prompt = f"""ข้อมูลสุขภาพของผู้ใช้:
- อายุ: {user['Age']} ปี
- BMI: {round(user['BMI'], 1)}
- คะแนนการนอน: {user['Sleep_Health_Score']}/100
- คะแนนการออกกำลังกาย: {user['Activity_Health_Score']}/100
- คะแนนหัวใจ: {user['Cardiovascular_Health_Score']}/100
- คะแนนสุขภาพจิต: {user['Mental_Health_Score']}/100
- คะแนนรวม: {round(user['Overall_Wellness_Score'], 1)}/100

กรุณาให้คำแนะนำ:
1. 🏋️ แผนออกกำลังกาย
2. 🍎 แผนโภชนาการ
3. 😴 การนอนและสุขภาพจิต
4. ⚠️ ความเสี่ยงที่ควรระวัง"""

    return {"recommendation": generate([{"role": "user", "content": prompt}])}

@app.post("/symptom-check")
async def symptom_check(data: dict):
    symptoms = data.get("symptoms", "")
    age = data.get("age", "ไม่ระบุ")
    prompt = f"""ผู้ป่วยอายุ {age} ปี มีอาการ: {symptoms}

วิเคราะห์:
1. 🔍 การวิเคราะห์อาการ
2. ⚠️ ระดับความรุนแรง
3. 💊 การดูแลเบื้องต้น
4. 🏥 ควรพบแพทย์เมื่อไหร่
5. 🌿 คำแนะนำเพิ่มเติม

⚠️ นี่เป็นการวิเคราะห์เบื้องต้นเท่านั้น"""

    return {"result": generate([{"role": "user", "content": prompt}])}

@app.post("/chat")
async def chat(data: dict):
    messages = data.get("messages", [])
    return {"reply": generate(messages)}

@app.post("/analyze-food")
async def analyze_food(data: dict):
    image_base64 = data.get("image", "")

    prompt = """วิเคราะห์อาหารในรูปนี้ครับ ให้ข้อมูลดังนี้:

🍽️ **ชื่ออาหาร** และปริมาณโดยประมาณ

📊 **ข้อมูลโภชนาการ (ต่อ 1 จาน)**
- 🔥 แคลอรี่: XX kcal
- 🥩 โปรตีน: XXg
- 🍚 คาร์โบไฮเดรต: XXg
- 🧈 ไขมัน: XXg
- 🧂 โซเดียม: XXmg
- 🌾 ใยอาหาร: XXg

✅ **ข้อดี** ของอาหารนี้

⚠️ **ข้อควรระวัง**

💡 **คำแนะนำ** สำหรับคนที่ต้องการควบคุมน้ำหนัก/สุขภาพ

ตอบเป็นภาษาไทย กระชับ เข้าใจง่ายครับ
ถ้าไม่ใช่รูปอาหาร ให้บอกว่า "ไม่พบอาหารในรูปนี้ครับ" """

    image_data = base64.b64decode(image_base64)

    response = vision_model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": image_data}
    ])

    return {"result": response.text}