import streamlit as st
import pandas as pd
import os
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "students.csv"

# Change this password to your own password
ADMIN_PASSWORD = "admin123"


# ============================================================
# FUNCTIONS
# ============================================================

def save_student_data(name, email, phone, branch, course, year):
    """
    Save one student's information into students.csv.
    """

    new_student = pd.DataFrame({
        "Date & Time": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ],
        "Name": [name],
        "Gmail": [email],
        "Phone": [phone],
        "Branch": [branch],
        "Course": [course],
        "Year": [year]
    })

    # If file exists, append new student
    if os.path.exists(DATA_FILE):

        new_student.to_csv(
            DATA_FILE,
            mode="a",
            header=False,
            index=False
        )

    # Otherwise create new file
    else:

        new_student.to_csv(
            DATA_FILE,
            mode="w",
            header=True,
            index=False
        )


def load_student_data():
    """
    Load all student records.
    """

    if os.path.exists(DATA_FILE):

        return pd.read_csv(DATA_FILE)

    return pd.DataFrame(
        columns=[
            "Date & Time",
            "Name",
            "Gmail",
            "Phone",
            "Branch",
            "Course",
            "Year"
        ]
    )


# ============================================================
# GET STUDENT DATA FROM HTML
# ============================================================

params = st.query_params

name = params.get("name", "")
email = params.get("email", "")
phone = params.get("phone", "")
branch = params.get("branch", "")
course = params.get("course", "")
year = params.get("year", "")


# ============================================================
# MAIN TITLE
# ============================================================

st.title("🎓 Student Performance Prediction System")

st.write(
    "Welcome to the Student Performance Prediction System."
)


# ============================================================
# SAVE STUDENT DATA
# ============================================================

if name and email and phone and branch and course and year:

    # Create unique session key so the same URL refresh
    # does not repeatedly save the same student.
    current_student = (
        f"{name}|{email}|{phone}|{branch}|{course}|{year}"
    )

    if st.session_state.get("saved_student") != current_student:

        save_student_data(
            name,
            email,
            phone,
            branch,
            course,
            year
        )

        st.session_state.saved_student = current_student

        st.success(
            "✅ Your information has been saved successfully!"
        )


# ============================================================
# STUDENT INFORMATION
# ============================================================

if name:

    st.header("👨‍🎓 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Student Name**")
        st.info(name)

        st.write("**Gmail**")
        st.info(email)

        st.write("**Phone Number**")
        st.info(phone)

    with col2:

        st.write("**Branch**")
        st.info(branch)

        st.write("**Course**")
        st.info(course)

        st.write("**Year**")
        st.info(year)


else:

    st.info(
        "Please open your HTML login page and enter "
        "your student information."
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header("📊 Student Performance Prediction")

st.write(
    "Enter the academic information required for prediction."
)


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

    assignment_completion = st.number_input(
        "Assignment Completion (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )


# ============================================================
# SIMPLE DEMO PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Performance",
    type="primary"
):

    # Simple demo calculation
    prediction = (
        attendance * 0.30
        + study_hours * 5
        + previous_score * 0.30
        + assignment_completion * 0.20
    )

    prediction = min(
        max(prediction, 0),
        100
    )

    st.subheader("📈 Prediction Result")

    st.metric(
        "Predicted Performance",
        f"{prediction:.2f}%"
    )

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
# ADMIN RECORDS
# ============================================================

st.divider()

st.header("🔒 Admin Records")

st.write(
    "This section is only for the administrator."
)


# Password input
admin_password = st.text_input(
    "Enter Admin Password",
    type="password"
)


if st.button("🔐 Login to Admin Records"):

    if admin_password == ADMIN_PASSWORD:

        st.session_state["admin_logged_in"] = True

        st.success(
            "✅ Admin login successful."
        )

    else:

        st.session_state["admin_logged_in"] = False

        st.error(
            "❌ Incorrect admin password."
        )


# ============================================================
# SHOW ADMIN RECORDS
# ============================================================

if st.session_state.get("admin_logged_in", False):

    st.subheader("📋 All Student Records")

    students = load_student_data()


    if len(students) > 0:

        # Number of students
        st.metric(
            "Total Students",
            len(students)
        )


        # Display records
        st.dataframe(
            students,
            use_container_width=True,
            hide_index=True
        )


        # Convert data to CSV
        csv_data = students.to_csv(
            index=False
        ).encode("utf-8")


        # Download button
        st.download_button(
            label="⬇️ Download Student Records CSV",
            data=csv_data,
            file_name="student_records.csv",
            mime="text/csv"
        )


    else:

        st.info(
            "No student records have been saved yet."
        )


    # Admin logout
    if st.button("🚪 Logout Admin"):

        st.session_state["admin_logged_in"] = False

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "© 2026 Student Performance Prediction System"
)
