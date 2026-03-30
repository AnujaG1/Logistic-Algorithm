import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Pass/Fail Predictor", layout="wide")

st.markdown("""
<style>
.report-pass {
    background: linear-gradient(135deg, #1a472a, #2ecc71);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    color: white;
    margin-bottom: 16px;
}
.report-fail {
    background: linear-gradient(135deg, #6b1a1a, #e74c3c);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    color: white;
    margin-bottom: 16px;
}
.report-title {
    font-size: 13px;
    letter-spacing: 3px;
    opacity: 0.85;
    margin-bottom: 8px;
}
.report-result {
    font-size: 52px;
    font-weight: 900;
    letter-spacing: 8px;
    margin: 6px 0;
}
.report-prob {
    font-size: 17px;
    opacity: 0.9;
}
.section-label {
    font-size: 13px;
    font-weight: 600;
    color: #888;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin: 20px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL FILES ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    theta           = np.load('logistic_weights.npy')
    mean            = np.load('scaler_mean.npy')
    std             = np.load('scaler_std.npy')
    feature_columns = pd.read_csv('X_features.csv').columns.tolist()
    return theta, mean, std, feature_columns

theta, mean, std, feature_columns = load_model()

# ── TITLE ────────────────────────────────────────────────────────
st.title("🎓 Student Performance Predictor")
st.markdown("**Logistic Regression from Scratch** – Predict if a student will **Pass** or **Fail**")
st.markdown("---")

# ── TWO COLUMN LAYOUT ────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ════════════════════════════════════
# LEFT — INPUTS (exactly your original)
# ════════════════════════════════════
with left:
    st.subheader("Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        age                   = st.slider("Age", 15, 20, 17)
        study_hours           = st.slider("Study Hours per Day", 0.0, 15.0, 5.0, step=0.5)
        attendance            = st.slider("Attendance (%)", 50, 100, 85)
        sleep_hours           = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.0, step=0.5)
        previous_grade        = st.slider("Previous Grade (%)", 40, 100, 75)
        assignments_completed = st.slider("Assignments Completed", 0, 20, 12)
        practice_tests_taken  = st.slider("Practice Tests Taken", 0, 10, 4)

    with col2:
        group_study_hours     = st.slider("Group Study Hours", 0.0, 10.0, 2.0, step=0.5)
        notes_quality_score   = st.slider("Notes Quality Score (1-10)", 1, 10, 7)
        time_management_score = st.slider("Time Management Score (1-10)", 1, 10, 7)
        motivation_level      = st.slider("Motivation Level (1-10)", 1, 10, 7)
        mental_health_score   = st.slider("Mental Health Score (1-10)", 1, 10, 8)
        screen_time           = st.slider("Screen Time (hours/day)", 0.0, 10.0, 4.0, step=0.5)
        social_media_hours    = st.slider("Social Media Hours (per day)", 0.0, 8.0, 2.0, step=0.5)

    gender           = st.selectbox("Gender", ["Male", "Female"])
    parent_education = st.selectbox("Parent Education", ["High School", "Some College", "Bachelor's", "Master's", "PhD"])
    internet_access  = st.selectbox("Internet Access", ["Yes", "No"])
    device_type      = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet", "None"])
    school_type      = st.selectbox("School Type", ["Public", "Private"])
    extracurriculars = st.selectbox("Extracurriculars", ["None", "Coding Club", "Music", "Debate", "Sports", "Arts"])
    family_income    = st.selectbox("Family Income", ["Low", "Medium", "High"])

    predict_btn = st.button("🚀 Predict Pass/Fail", type="primary", use_container_width=True)

# ════════════════════════════════════
# RIGHT — REPORT CARD
# ════════════════════════════════════
with right:
    st.subheader("📄 Report Card")

    if not predict_btn:
        st.info("👈 Fill in the student details and click **Predict Pass/Fail** to generate the report card.")

    else:
        # ── Prediction (same logic as your original) ──────────────
        input_data = pd.DataFrame({
            'age': [age], 'gender': [gender], 'study_hours': [study_hours],
            'attendance': [attendance], 'sleep_hours': [sleep_hours],
            'previous_grade': [previous_grade],
            'assignments_completed': [assignments_completed],
            'practice_tests_taken': [practice_tests_taken],
            'group_study_hours': [group_study_hours],
            'notes_quality_score': [notes_quality_score],
            'time_management_score': [time_management_score],
            'motivation_level': [motivation_level],
            'mental_health_score': [mental_health_score],
            'screen_time': [screen_time],
            'social_media_hours': [social_media_hours],
            'family_income': [family_income],
            'parent_education': [parent_education],
            'internet_access': [internet_access],
            'device_type': [device_type],
            'school_type': [school_type],
            'extracurriculars': [extracurriculars]
        })

        categorical_cols = ['gender', 'parent_education', 'internet_access', 'device_type',
                            'school_type', 'extracurriculars', 'family_income']

        input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        input_scaled  = (input_encoded.values - mean) / std
        input_scaled  = np.c_[np.ones(1), input_scaled]

        probability = 1 / (1 + np.exp(-input_scaled @ theta))[0]
        prediction  = "PASS" if probability >= 0.5 else "FAIL"

        # ── Result banner ──────────────────────────────────────────
        card_class = "report-pass" if prediction == "PASS" else "report-fail"
        emoji      = "🎉" if prediction == "PASS" else "❌"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="report-title">STUDENT REPORT CARD · IOE PULCHOWK</div>
            <div class="report-result">{emoji} {prediction}</div>
            <div class="report-prob">Pass Probability: <strong>{probability*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

        if prediction == "PASS":
            st.balloons()

        # ── Key metrics ────────────────────────────────────────────
        st.markdown('<div class="section-label">Key Inputs</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Study Hours",    f"{study_hours} hrs")
        m2.metric("Attendance",     f"{attendance}%")
        m3.metric("Previous Grade", f"{previous_grade}%")

        m4, m5, m6 = st.columns(3)
        m4.metric("Assignments",    f"{assignments_completed}/20")
        m5.metric("Practice Tests", f"{practice_tests_taken}/10")
        m6.metric("Motivation",     f"{motivation_level}/10")

        # ── Probability bar ────────────────────────────────────────
        st.markdown('<div class="section-label">Pass Probability</div>', unsafe_allow_html=True)
        st.progress(float(probability))
        st.caption(f"Threshold: 0.50 — Student probability: {probability:.4f}")

        # ── Personalized tips ──────────────────────────────────────
        st.markdown('<div class="section-label">Personalized Tips</div>', unsafe_allow_html=True)

        tips = []
        if study_hours < 4:
            tips.append("📚 Study hours are low. Aim for at least 5 hrs/day.")
        if attendance < 80:
            tips.append("🏫 Attendance below 80% significantly hurts performance.")
        if practice_tests_taken < 3:
            tips.append("📋 Take more practice tests to build exam confidence.")
        if screen_time > 5:
            tips.append("📱 High screen time. Try limiting to under 4 hrs/day.")
        if social_media_hours > 3:
            tips.append("📲 Reduce social media usage during study hours.")
        if motivation_level < 5:
            tips.append("🎯 Low motivation. Set small daily goals to stay on track.")
        if sleep_hours < 6:
            tips.append("😴 Sleep is too low. Aim for at least 7 hrs per night.")
        if assignments_completed < 8:
            tips.append("📝 Complete more assignments — they directly reflect exam readiness.")

        if tips:
            for tip in tips:
                st.warning(tip)
        else:
            st.success("✅ Excellent profile! Keep up the good habits.")