import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Predict student performance using machine learning."
)


# ============================================================
# READ DATA FROM HTML URL
# ============================================================

params = st.query_params


name = params.get("name", "")
email = params.get("email", "")
phone = params.get("phone", "")
branch = params.get("branch", "")
course = params.get("course", "")
year = params.get("year", "")


# ============================================================
# STUDENT INFORMATION
# ============================================================

if name:

    st.success(
        f"Welcome, {name}! Your details have been received."
    )

    st.subheader("👨‍🎓 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Student Name:**")
        st.info(name)

        st.write("**Gmail:**")
        st.info(email)

        st.write("**Phone Number:**")
        st.info(phone)

    with col2:

        st.write("**Branch:**")
        st.info(branch)

        st.write("**Course:**")
        st.info(course)

        st.write("**Year:**")
        st.info(year)

    st.divider()


else:

    st.info(
        "No student information was received. "
        "Please open the HTML login page first."
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.header("📊 Performance Prediction")

st.write(
    "Enter the academic information required by your "
    "machine-learning model."
)


# ============================================================
# EXAMPLE PREDICTION INPUTS
# ============================================================
#
# IMPORTANT:
# Replace these inputs with the EXACT features
# used by your trained ML model.
#

col1, col2 = st.columns(2)


with col1:

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=3.0
    )


with col2:

    previous_score = st.number_input(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

    assignments = st.number_input(
        "Assignments Completed (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )


# ============================================================
# SAMPLE MACHINE LEARNING MODEL
# ============================================================
#
# This section is only an example.
#
# If you already have your dataset/model,
# use your existing training code instead.
# ============================================================


def create_demo_model():

    # Demo training data
    data = {

        "attendance": [
            50, 55, 60, 65, 70,
            75, 80, 85, 90, 95
        ],

        "study_hours": [
            1, 1.5, 2, 2.5, 3,
            3.5, 4, 5, 6, 7
        ],

        "previous_score": [
            40, 45, 50, 55, 60,
            65, 70, 75, 85, 90
        ],

        "assignments": [
            40, 45, 50, 60, 65,
            70, 75, 80, 90, 95
        ],

        "performance": [
            42, 47, 52, 58, 63,
            68, 73, 78, 88, 94
        ]
    }


    df = pd.DataFrame(data)


    X = df[
        [
            "attendance",
            "study_hours",
            "previous_score",
            "assignments"
        ]
    ]

    y = df["performance"]


    model = LinearRegression()

    model.fit(X, y)

    return model


model = create_demo_model()


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🔮 Predict Student Performance",
    type="primary"
):

    input_data = pd.DataFrame({

        "attendance": [attendance],

        "study_hours": [study_hours],

        "previous_score": [previous_score],

        "assignments": [assignments]

    })


    prediction = model.predict(input_data)[0]


    # Keep prediction between 0 and 100
    prediction = max(
        0,
        min(100, prediction)
    )


    st.divider()

    st.subheader(
        "📈 Prediction Result"
    )


    st.metric(
        "Predicted Performance",
        f"{prediction:.2f}%"
    )


    # Performance category

    if prediction >= 75:

        st.success(
            "🌟 Excellent Performance"
        )

    elif prediction >= 60:

        st.info(
            "👍 Good Performance"
        )

    elif prediction >= 40:

        st.warning(
            "⚠️ Average Performance"
        )

    else:

        st.error(
            "📚 Needs Improvement"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Machine Learning Project"
)
