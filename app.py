import streamlit as st
import pandas as pd
import pickle

# Title
st.title("🎓 Student Performance Prediction")
st.write("Predict a student’s final grade based on academic and social factors.")

# Sidebar input form
st.sidebar.header("Enter Student Details")

study_time = st.sidebar.slider("Study Time (hours per day)", 0, 10, 2)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
previous_grade = st.sidebar.slider("Previous Grade (%)", 0, 100, 70)
family_support = st.sidebar.selectbox("Family Support", ["Low", "High"])
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["No", "Yes"])
health = st.sidebar.slider("Health (1=Poor, 5=Excellent)", 1, 5, 3)

# Convert categorical features to numeric
family_support_map = {"Low": 0, "High": 1}
extracurricular_map = {"No": 0, "Yes": 1}

data = {
    "study_time": study_time,
    "attendance": attendance,
    "previous_grade": previous_grade,
    "family_support": family_support_map[family_support],
    "extracurricular": extracurricular_map[extracurricular],
    "health_feature": health
}

input_df = pd.DataFrame([data])

# Load model and scaler
try:
    with open("student_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Scale input data
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    st.subheader("📊 Predicted Final Grade:")
    st.success(f"{prediction:.2f} / 20")

except FileNotFoundError:
    st.error("⚠️ Model or scaler file not found. Please train your model again using train_model_for_app.py.")
