import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Student Pass/Fail Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")
st.markdown("**Logistic Regression from Scratch** – Predict if a student will **Pass** or **Fail**")

# LOAD MODEL FILES
@st.cache_resource
def load_model():
    theta = np.load('logistic_weights.npy')
    mean = np.load('scaler_mean.npy')
    std = np.load('scaler_std.npy')
    feature_columns = pd.read_csv('X_features.csv').columns.tolist()
    return theta, mean, std, feature_columns

theta, mean, std, feature_columns = load_model()

#USER INPUTS
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 20, 17)
    study_hours = st.slider("Study Hours per Day", 0.0, 15.0, 5.0, step=0.5)
    attendance = st.slider("Attendance (%)", 50, 100, 85)
    sleep_hours = st.slider("Sleep Hours per Night", 4.0, 12.0, 7.0, step=0.5)
    previous_grade = st.slider("Previous Grade (%)", 40, 100, 75)
    assignments_completed = st.slider("Assignments Completed", 0, 20, 12)
    practice_tests_taken = st.slider("Practice Tests Taken", 0, 10, 4)

with col2:
    group_study_hours = st.slider("Group Study Hours", 0.0, 10.0, 2.0, step=0.5)
    notes_quality_score = st.slider("Notes Quality Score (1-10)", 1, 10, 7)
    time_management_score = st.slider("Time Management Score (1-10)", 1, 10, 7)
    motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 7)
    mental_health_score = st.slider("Mental Health Score (1-10)", 1, 10, 8)
    screen_time = st.slider("Screen Time (hours/day)", 0.0, 10.0, 4.0, step=0.5)
    social_media_hours = st.slider("Social Media Hours (per day)", 0.0, 8.0, 2.0, step=0.5)

# Categorical inputs
gender = st.selectbox("Gender", ["Male", "Female"])
parent_education = st.selectbox("Parent Education", ["High School", "Some College", "Bachelor's", "Master's", "PhD"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
device_type = st.selectbox("Device Type", ["Mobile", "Laptop", "Tablet", "None"])
school_type = st.selectbox("School Type", ["Public", "Private"])
extracurriculars = st.selectbox("Extracurriculars", ["None", "Coding Club", "Music", "Debate", "Sports", "Arts"])
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])

#PREDICT BUTTON
if st.button("🚀 Predict Pass/Fail", type="primary"):
    # Create single-row DataFrame (same format as original)
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'study_hours': [study_hours],
        'attendance': [attendance],
        'sleep_hours': [sleep_hours],
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

    # One-hot encode exactly like in training
    categorical_cols = ['gender', 'parent_education', 'internet_access', 'device_type',
                        'school_type', 'extracurriculars', 'family_income']
    
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Align columns exactly with training data (fill missing dummies with 0)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale using same scaler
    input_scaled = (input_encoded.values - mean) / std

    # Add bias term
    input_scaled = np.c_[np.ones(1), input_scaled]

    # Predict
    probability = 1 / (1 + np.exp(-input_scaled @ theta))[0]
    prediction = "PASS" if probability >= 0.5 else "FAIL"

    # Show result
    st.success(f"**Prediction: {prediction}**")
    st.metric("Probability of Passing", f"{probability*100:.1f}%")

    # Confidence bar
    st.progress(float(probability))
    
    if prediction == "PASS":
        st.balloons()