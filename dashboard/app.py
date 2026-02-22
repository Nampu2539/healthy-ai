import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Health AI Dashboard", layout="wide")

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("../data/output/health_recommendations.csv")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("🧭 Navigation")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

template = "plotly_dark" if dark_mode else "plotly_white"

page = st.sidebar.radio("", 
["👤 Profile","📊 Analytics","🤖 Recommendations","📘 Model Info"])

user_id = st.sidebar.slider("Select User",0,len(df)-1,0)
user = df.iloc[user_id]

# ----------------------------
# DARK SIDEBAR STYLE
# ----------------------------
if dark_mode:
    st.markdown("""
    <style>

    /* ===== APP BACKGROUND ===== */
    .stApp {
        background-color: #0B1220;  /* navy gray */
        color: #E6EDF3;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #E6EDF3 !important;
    }

    /* ===== TITLES ===== */
    h1,h2,h3,h4 {
        color: #F9FAFB !important;
        font-weight:600;
    }

    /* ===== TEXT ===== */
    p,span,label {
        color: #CBD5E1 !important;
    }

    /* ===== METRIC CARDS ===== */
    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    /* ===== NORMAL CARDS ===== */
    .card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 6px 30px rgba(0,0,0,0.25);
    }

    /* ===== SLIDER ===== */
    .stSlider > div > div > div > div {
        background: #22C55E !important;
    }

    /* ===== TOGGLE ===== */
    .stToggle > div {
        background-color: #22C55E !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background:#111827;
        border-radius:12px;
    }

    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.title("🏥 Health & Wellness AI Dashboard")
st.caption("AI-Powered Lifestyle Recommendation System")

# ======================================================
# 👤 PROFILE PAGE
# ======================================================
if page == "👤 Profile":

    st.subheader("👤 User Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Age", user["Age"])
    c2.metric("BMI", round(user["BMI"],1))
    c3.metric("Wellness Score", round(user["Overall_Wellness_Score"],1))
    c4.metric("Segment", user["User_Segment"])

    # -------- RISK BADGE --------
    score = user["Overall_Wellness_Score"]
    if score < 50:
        st.error("🚨 High Risk – Immediate Attention Needed")
    elif score < 70:
        st.warning("⚠️ Medium Risk – Monitor Lifestyle")
    else:
        st.success("✅ Low Risk – Healthy")

    # -------- HEALTH BAR --------
    st.subheader("💪 Health Metrics")
    scores = {
        "Sleep": user["Sleep_Health_Score"],
        "Activity": user["Activity_Health_Score"],
        "Cardio": user["Cardiovascular_Health_Score"],
        "Mental": user["Mental_Health_Score"]
    }

    fig = px.bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        color=list(scores.values()),
        color_continuous_scale="RdYlGn",
        template=template
    )
    st.plotly_chart(fig, width="stretch")

    # -------- GAUGE --------
    st.subheader("💯 Overall Wellness")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range':[0,100]},
               'bar': {'color': "#22C55E"},
               'steps':[
                   {'range':[0,40],'color':"#7F1D1D"},
                   {'range':[40,70],'color':"#92400E"},
                   {'range':[70,100],'color':"#14532D"}]
               }))
    gauge.update_layout(template=template)
    st.plotly_chart(gauge, width="stretch")

    # -------- YOU VS POPULATION --------
    st.subheader("📊 You vs Population")
    compare = pd.DataFrame({
        "Category":["You","Population Avg"],
        "Score":[score, df["Overall_Wellness_Score"].mean()]
    })
    fig_compare = px.bar(compare,x="Category",y="Score",template=template)
    st.plotly_chart(fig_compare,width="stretch")

# ======================================================
# 📊 ANALYTICS PAGE
# ======================================================
if page == "📊 Analytics":

    st.subheader("📊 Population Analytics")

    col1,col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df,x="Overall_Wellness_Score",
                            nbins=25,template=template)
        st.plotly_chart(fig1,width="stretch")

    with col2:
        fig2 = px.pie(df,names="User_Segment",template=template)
        st.plotly_chart(fig2,width="stretch")

    # -------- INSIGHT --------
    avg_sleep = df["Sleep_Health_Score"].mean()
    st.info(f"📈 Insight: Avg Sleep Score = {avg_sleep:.1f}. "
            "Higher sleep correlates with higher wellness.")

# ======================================================
# 🤖 RECOMMENDATIONS PAGE
# ======================================================
if page == "🤖 Recommendations":

    st.subheader("🤖 Personalized Recommendations")

    st.success("🏋️ Exercise Plan")
    st.info(user["Exercise_Recommendation"])

    st.success("🍎 Nutrition Plan")
    st.info(user["Nutrition_Recommendation"])

    st.success("😴 Lifestyle Advice")
    st.info(user["Lifestyle_Recommendation"])

    # -------- EXPLAINABLE AI --------
    with st.expander("🔍 Why this recommendation?"):
        st.write(f"""
        Sleep Score: {user["Sleep_Health_Score"]}
        Activity Score: {user["Activity_Health_Score"]}
        BMI: {round(user["BMI"],1)}

        Recommendation generated using rule-based AI
        based on health score thresholds.
        """)

    # -------- DOWNLOAD REPORT --------
    st.download_button(
        "⬇️ Download User Report",
        data=user.to_csv(),
        file_name=f"user_{user_id}_report.csv"
    )

# ======================================================
# 📘 MODEL INFO PAGE
# ======================================================
if page == "📘 Model Info":

    st.subheader("📘 AI Model Overview")

    st.markdown("""
    **Pipeline**
    - Data Cleaning → Feature Engineering  
    - KMeans Clustering → User Segmentation  
    - Rule-Based Recommendation Engine  
    - Streamlit Dashboard  

    **Tech Stack**
    - Python / Pandas / Scikit-Learn  
    - Plotly Visualization  
    - Streamlit Deployment Ready  
    """)

st.markdown("---")
st.caption("© 2026 Health AI • Production Ready")
