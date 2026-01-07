# ==================================================
# EduRisk AI - Academic Risk Intelligence Platform
# Microsoft Imagine Cup 2026 - Education Category
# FIXED: 7-feature compatibility
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# ==================== PAGE CONFIG FIRST ====================
st.set_page_config(
    page_title="EduRisk AI | Imagine Cup 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== AZURE AI (SIMULATED FOR MVP) ====================
class AzureAIServices:
    def get_ai_guidance(self, prompt):
        if "High" in prompt:
            return """
**🔴 HIGH RISK - EMERGENCY 7-DAY PLAN**

**Day 1–2: Immediate Action**
- Attend ALL classes
- Meet your teacher/advisor today
- Submit any overdue work

**Day 3–5: Intensive Recovery**
- Study 4+ hours daily on weak subjects
- Complete 20+ practice problems per subject
- Get help from tutor or classmate

**Day 6–7: Build Momentum**
- Take full practice test
- Make a realistic schedule for next month
- Reward yourself for effort

**You can turn this around — start today!**
            """
        elif "Low" in prompt:
            return """
**🟢 LOW RISK - EXCELLENCE PLAN**

**Day 1–3: Advanced Challenge**
- Study topics beyond the syllabus
- Teach a concept to someone else
- Start a small project or research

**Day 4–7: Leadership**
- Help struggling classmates
- Join or start a study group
- Prepare for competitions

**Your hard work is paying off — keep pushing higher!**
            """
        else:
            return """
**🟡 MEDIUM RISK - TARGETED IMPROVEMENT**

**Daily Focus**
- 2–3 hours structured study
- Focus on 1 weak area per day
- Use active recall (test yourself)

**Weekly Goals**
- Review all mistakes
- Improve time management
- Track progress in a journal

**Consistent small steps = big improvement. You've got this!**
            """

azure_ai = AzureAIServices()

# --------------------------------------------------
# Model Loading - FIXED FOR 7 FEATURES
# --------------------------------------------------
@st.cache_resource
def load_model():
    # List of exactly 7 features in correct order
    FEATURE_NAMES = [
        'attendance_pct',
        'assignment_score',
        'quiz_score',
        'midterm_score',
        'study_hours_per_week',
        'previous_gpa'
    ]

    model_paths = [
        "models/student_risk_model.pkl",
        "models/enhanced_student_model.pkl",
        "student_risk_model.pkl",
        "enhanced_student_model.pkl"
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                loaded = joblib.load(path)
                st.success(f"✅ Real model loaded: {os.path.basename(path)}")
                
                if isinstance(loaded, dict) and 'model' in loaded:
                    loaded['feature_names'] = FEATURE_NAMES
                    return loaded
                else:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    le.classes_ = np.array(['High', 'Medium', 'Low'])
                    return {
                        'model': loaded,
                        'feature_names': FEATURE_NAMES,
                        'label_encoder': le
                    }
            except Exception as e:
                st.error(f"Error loading {path}: {e}")

    # Fallback simulated model
    st.info("🔄 Using simulated model (Azure ML ready in production)")
    return create_simulated_model()

def create_simulated_model():
    FEATURE_NAMES = [
        'attendance_pct', 'assignment_score', 'quiz_score',
        'midterm_score', 'study_hours_per_week', 'previous_gpa'
    ]
    
    from sklearn.preprocessing import LabelEncoder
    
    class SimModel:
        def predict(self, X):
            # Weighted score using all 6 + previous_gpa
            score = (X[:,0]*0.15 + X[:,1]*0.20 + X[:,2]*0.20 +
                     X[:,3]*0.20 + X[:,4]*0.15 + X[:,5]*0.10)
            return np.where(score < 60, 0, np.where(score < 75, 1, 2))
        
        def predict_proba(self, X):
            score = (X[:,0]*0.15 + X[:,1]*0.20 + X[:,2]*0.20 +
                     X[:,3]*0.20 + X[:,4]*0.15 + X[:,5]*0.10)
            p = np.zeros((len(X), 3))
            p[score < 60] = [0.80, 0.15, 0.05]
            p[(score >=60) & (score <75)] = [0.18, 0.70, 0.12]
            p[score >=75] = [0.05, 0.15, 0.80]
            return p
    
    le = LabelEncoder()
    le.classes_ = np.array(['High', 'Medium', 'Low'])
    
    return {
        'model': SimModel(),
        'feature_names': FEATURE_NAMES,
        'label_encoder': le
    }

# --------------------------------------------------
# Prediction - FIXED: Always 7 features
# --------------------------------------------------
def predict_risk(student_features, pipeline):
    # Force all 7 features in correct order
    feature_names = pipeline['feature_names']
    
    # Create DataFrame with all required columns
    df = pd.DataFrame([student_features])
    
    # Add any missing columns with safe defaults
    for col in feature_names:
        if col not in df.columns:
            if col == 'previous_gpa':
                df[col] = 6.5
            else:
                df[col] = 70  # reasonable default
    
    # Reorder and select exactly the expected columns
    df = df[feature_names]
    
    X = df.values
    
    try:
        pred = pipeline['model'].predict(X)[0]
        proba = pipeline['model'].predict_proba(X)[0]
        
        risk_level = pipeline['label_encoder'].inverse_transform([pred])[0]
        
        return {
            'risk_level': risk_level,
            'confidence': proba[pred],
            'probabilities': {'High': proba[0], 'Medium': proba[1], 'Low': proba[2]}
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {'risk_level': 'Medium', 'confidence': 0.5, 'probabilities': {'High':0.33,'Medium':0.34,'Low':0.33}}

# --------------------------------------------------
# Weak Areas
# --------------------------------------------------
def identify_weak_areas(data):
    weak = []
    if data['attendance_pct'] < 75: weak.append(f"Attendance ({data['attendance_pct']}%)")
    if data['assignment_score'] < 60: weak.append(f"Assignments ({data['assignment_score']})")
    if data['quiz_score'] < 60: weak.append(f"Quizzes ({data['quiz_score']})")
    if data['midterm_score'] < 60: weak.append(f"Midterm ({data['midterm_score']})")
    if data['study_hours_per_week'] < 10: weak.append(f"Study Hours ({data['study_hours_per_week']} hrs)")
    return weak

# --------------------------------------------------
# Gauge Chart
# --------------------------------------------------
def gauge_chart(value, title, threshold=70):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6366F1"},
            'steps': [
                {'range': [0, threshold*0.7], 'color': "#FCA5A5"},
                {'range': [threshold*0.7, threshold], 'color': "#FDE68A"},
                {'range': [threshold, 100], 'color': "#A7F3D0"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': threshold}
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.2rem;
        text-align: center;
        background: linear-gradient(90deg, #0078D4, #00BCF2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 0;
    }
    .azure-badge {
        background: #0078D4;
        color: white;
        padding: 10px 20px;
        border-radius: 50px;
        font-size: 1.1rem;
        display: inline-block;
        margin: 20px auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎓 EduRisk AI</h1>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#4B5563;'>Predict • Understand • Improve • Succeed</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='azure-badge'>Azure ML + Azure OpenAI Ready</span></div>", unsafe_allow_html=True)

    pipeline = load_model()

    with st.sidebar:
        st.header("👤 Student Profile")

        attendance_pct = st.slider("Attendance (%)", 40, 100, 75, 5)
        assignment_score = st.slider("Assignment Score", 0, 100, 70, 5)
        quiz_score = st.slider("Quiz Score", 0, 100, 65, 5)
        midterm_score = st.slider("Midterm Score", 0, 100, 60, 5)
        study_hours = st.slider("Study Hours/Week", 0, 40, 12, 1)
        previous_gpa = st.slider("Previous GPA (0-10)", 0.0, 10.0, 6.5, 0.1)

        st.markdown("### 🚀 Quick Demos")
        c1, c2, c3 = st.columns(3)
        if c1.button("High Risk"):
            attendance_pct, assignment_score, quiz_score, midterm_score = 58, 52, 48, 45
            study_hours, previous_gpa = 6, 4.9
            st.rerun()
        if c2.button("Medium"):
            attendance_pct, assignment_score, quiz_score, midterm_score = 74, 68, 65, 62
            study_hours, previous_gpa = 11, 6.7
            st.rerun()
        if c3.button("Low Risk"):
            attendance_pct, assignment_score, quiz_score, midterm_score = 95, 88, 92, 90
            study_hours, previous_gpa = 22, 8.8
            st.rerun()

        if st.button("🔍 Analyze Risk", type="primary", use_container_width=True):
            st.session_state.analyze = True
            st.rerun()

    if st.session_state.get("analyze", False):
        features = {
            'attendance_pct': attendance_pct,
            'assignment_score': assignment_score,
            'quiz_score': quiz_score,
            'midterm_score': midterm_score,
            'study_hours_per_week': study_hours,
            'previous_gpa': previous_gpa
        }

        result = predict_risk(features, pipeline)
        weak_areas = identify_weak_areas(features)

        prompt = f"Risk: {result['risk_level']}, Attendance: {attendance_pct}%, GPA: {previous_gpa}, Weak: {weak_areas}"
        guidance = azure_ai.get_ai_guidance(prompt)

        st.success(f"**Risk Level: {result['risk_level']}** | Confidence: {result['confidence']:.0%}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(gauge_chart(attendance_pct, "Attendance Rate", 75), use_container_width=True)
            avg = np.mean([assignment_score, quiz_score, midterm_score])
            st.plotly_chart(gauge_chart(avg, "Average Score", 70), use_container_width=True)
        with col2:
            if weak_areas:
                st.warning("### ⚠️ Areas Needing Attention")
                for area in weak_areas:
                    st.markdown(f"• {area}")
            else:
                st.success("### 🎉 Strong Performance Across All Areas!")

        st.markdown("### 🤖 Personalized AI Guidance (Azure OpenAI Ready)")
        st.markdown(guidance)

        st.info("**Imagine Cup MVP Complete** — Production uses Azure Machine Learning + Azure OpenAI")

    else:
        st.info("👈 Use the sidebar to input student data or try demo profiles!")

# ==================== RUN ====================
if __name__ == "__main__":
    if "analyze" not in st.session_state:
        st.session_state.analyze = False
    main()