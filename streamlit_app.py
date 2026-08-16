import streamlit as st
import pandas as pd
import os
from datetime import datetime


st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)


st.title("🎓 Student Performance Prediction System")


# =========================================================
# GET DATA FROM HTML
# =========================================================

params = st.query_params


name = params.get("name", "")
email = params.get("email", "")
phone = params.get("phone", "")
branch = params.get("branch", "")
course = params.get("course", "")
year = params.get("year", "")


# =========================================================
# DISPLAY STUDENT DATA
# =========================================================

if name:

    st.success(
        f"Welcome {name}! Your information has been received."
    )

    st.subheader("👨‍🎓 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Student Name")
        st.write(name)

        st.write("### Gmail")
        st.write(email)

        st.write("### Phone Number")
        st.write(phone)


    with col2:

        st.write("### Branch")
        st.write(branch)

        st.write("### Course")
        st.write(course)

        st.write("### Year")
        st.write(year)


    # =====================================================
    # SAVE DATA
    # =====================================================

    file_name = "students.csv"


    student_data = pd.DataFrame({

        "Date": [
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ],

        "Name": [name],

        "Gmail": [email],

        "Phone": [phone],

        "Branch": [branch],

        "Course": [course],

        "Year": [year]

    })


    # If file already exists, add new data
    if os.path.exists(file_name):

        student_data.to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )

    else:

        student_data.to_csv(
            file_name,
            index=False
        )


    st.success(
        "✅ Student data saved successfully!"
    )


    # =====================================================
    # SHOW ALL SAVED DATA
    # =====================================================

    st.divider()

    st.subheader("📋 Saved Student Data")

    all_students = pd.read_csv(file_name)

    st.dataframe(
        all_students,
        use_container_width=True
    )


else:

    st.warning(
        "Please open the HTML login page and enter "
        "student information first."
    )
