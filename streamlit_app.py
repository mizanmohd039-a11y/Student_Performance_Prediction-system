import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)



MODEL_CANDIDATES = [
    "best_student_model.pkl",
    "student_performance_model.pkl",
    "student_performance_model.joblib",
]

PREPROCESSING_CANDIDATES = [
    "preprocessing_elements.pkl",
]

DATASET_CANDIDATES = [
    "student_data.csv",
    "student_performance_dataset.csv",
    "student_performance_data.csv",
]

HISTORY_FILE = "prediction_history.json"


# ============================================================
# THE 15 REAL FEATURES
# ============================================================

FEATURES = [
    "Gender",
    "Age",
    "Attendance",
    "Study_Hours",
    "Assignment_Marks",
    "Internal_Marks",
    "Previous_Semester_Marks",
    "Internet_Access",
    "Family_Support",
    "Extra_Curricular",
    "Sleep_Hours",
    "Daily_Screen_Time",
    "Project_Marks",
    "Quiz_Marks",
    "Class_Participation",
]


# ============================================================
# FIND FILE
# ============================================================

def find_file(candidates):
    for filename in candidates:
        if os.path.exists(filename):
            return filename
    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = find_file(MODEL_CANDIDATES)

    if model_path is None:
        return None, None, None, (
            "Model file not found. Expected one of: "
            + ", ".join(MODEL_CANDIDATES)
        )

    try:
        loaded = joblib.load(model_path)

        model = loaded
        model_name = os.path.basename(model_path)
        model_features = None
        metrics = {}

        # Your original project saves a dictionary containing:
        # model_name, model_object, features and metrics.
        if isinstance(loaded, dict):

            model = (
                loaded.get("model_object")
                or loaded.get("model")
                or loaded.get("estimator")
            )

            model_name = loaded.get(
                "model_name",
                os.path.basename(model_path)
            )

            model_features = loaded.get("features")

            metrics = loaded.get("metrics", {})

        if model is None:
            return None, None, None, "The model file was loaded but no model object was found."

        return model, model_name, model_features, metrics

    except Exception as e:
        return None, None, None, f"Could not load model: {e}"


# ============================================================
# LOAD PREPROCESSING
# ============================================================

@st.cache_resource
def load_preprocessing():
    preprocessing_path = find_file(PREPROCESSING_CANDIDATES)

    if preprocessing_path is None:
        return None

    try:
        return joblib.load(preprocessing_path)
    except Exception:
        return None


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    dataset_path = find_file(DATASET_CANDIDATES)

    if dataset_path is None:
        return None

    try:
        return pd.read_csv(dataset_path)
    except Exception:
        return None


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()

    df["Average_Academic_Score"] = (
        df["Assignment_Marks"]
        + df["Internal_Marks"]
        + df["Project_Marks"]
        + df["Quiz_Marks"]
    ) / 4.0

    df["Attendance_Percentage"] = (
        df["Attendance"] / 100.0
    )

    df["Study_Efficiency_Score"] = (
        df["Study_Hours"]
        / (df["Daily_Screen_Time"] + 0.5)
    )

    df["Performance_Index"] = (
        df["Internal_Marks"] * 0.6
        + df["Previous_Semester_Marks"] * 0.4
    )

    df["Assignment_Ratio"] = (
        df["Assignment_Marks"] / 100.0
    )

    df["Participation_Score"] = (
        df["Attendance"] * 0.5
        + df["Class_Participation"] * 0.5
    )

    return df


# ============================================================
# PREPARE INPUT FOR MODEL
# ============================================================

def prepare_input(input_df, expected_features):

    df = input_df.copy()

    # Add columns used by original preprocessing pipeline.
    df["Final_Exam_Marks"] = 0.0
    df["Performance"] = "Average"

    # Engineer same features used during training.
    df = engineer_features(df)

    preprocessing = load_preprocessing()

    # --------------------------------------------------------
    # If preprocessing_elements.pkl exists, use the exact
    # scaler and encoders from training.
    # --------------------------------------------------------

    if preprocessing is not None:

        scaler = preprocessing.get("scaler")
        encoders = preprocessing.get("encoders", {})
        numerical_cols = preprocessing.get("numerical_cols", [])

        # Fill numerical missing values.
        for col in numerical_cols:
            if col not in df.columns:
                df[col] = 0.0

        for col in numerical_cols:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0.0)

        # Scale numerical columns exactly like training.
        if scaler is not None and len(numerical_cols) > 0:

            scaled = scaler.transform(
                df[numerical_cols]
            )

            scaled_df = pd.DataFrame(
                scaled,
                columns=numerical_cols,
                index=df.index
            )

            for col in numerical_cols:
                df[col] = scaled_df[col]

        # Encode categorical columns exactly like training.
        for col, encoder in encoders.items():

            if col not in df.columns:
                continue

            values = df[col].astype(str)

            # Unknown category -> first known class
            values = values.map(
                lambda x: x
                if x in encoder.classes_
                else encoder.classes_[0]
            )

            df[col] = encoder.transform(values)

    # --------------------------------------------------------
    # Select the exact features expected by model.
    # --------------------------------------------------------

    if expected_features is not None:

        missing = [
            col
            for col in expected_features
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                "The following model features are missing: "
                + ", ".join(missing)
            )

        df = df[expected_features]

    else:

        # Fallback for models exposing feature_names_in_
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            df = df[expected]

        else:
            # Remove non-model columns.
            remove_cols = [
                "Student_ID",
                "Final_Exam_Marks",
                "Performance"
            ]

            df = df.drop(
                columns=[
                    c for c in remove_cols
                    if c in df.columns
                ],
                errors="ignore"
            )

    # Final numeric conversion.
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).fillna(0.0)

    return df


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def performance_category(score):

    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 50:
        return "Average"

    return "Needs Improvement"


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendations(data, score):

    recommendations = []

    if data["Attendance"] < 75:
        recommendations.append(
            "Attendance is below 75%. Try to attend more classes consistently."
        )
    elif data["Attendance"] < 85:
        recommendations.append(
            "Attendance is moderate. Improving attendance may help maintain academic progress."
        )

    if data["Study_Hours"] < 4:
        recommendations.append(
            "Consider increasing daily study time gradually."
        )

    if data["Assignment_Marks"] < 65:
        recommendations.append(
            "Focus on completing assignments carefully and on time."
        )

    if data["Internal_Marks"] < 60:
        recommendations.append(
            "Spend additional time preparing for internal assessments."
        )

    if data["Previous_Semester_Marks"] < 60:
        recommendations.append(
            "Review previous-semester topics to strengthen your foundation."
        )

    if data["Project_Marks"] < 65:
        recommendations.append(
            "Give more attention to project work and practical learning."
        )

    if data["Quiz_Marks"] < 60:
        recommendations.append(
            "Use short, regular revision sessions to improve quiz performance."
        )

    if data["Study_Hours"] < 5 and data["Daily_Screen_Time"] > 5:
        recommendations.append(
            "Try to balance recreational screen time with focused study time."
        )

    if data["Sleep_Hours"] < 6:
        recommendations.append(
            "Maintain a regular sleep schedule so you can stay focused during study."
        )

    if data["Class_Participation"] < 50:
        recommendations.append(
            "Participate more actively in class discussions and activities."
        )

    if not recommendations:
        recommendations.append(
            "Your current profile looks balanced. Continue your existing study habits."
        )

    return recommendations


# ============================================================
# SAVE HISTORY
# ============================================================

def save_prediction(data, prediction, category):

    record = {
        "timestamp": pd.Timestamp.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "predicted_marks": round(float(prediction), 2),
        "performance": category,
        "inputs": data,
    }

    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(record)

    history = history[-100:]

    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(
                history,
                f,
                indent=4
            )
    except Exception:
        pass


# ============================================================
# LOAD EVERYTHING
# ============================================================

model, model_name, model_features, model_info = load_model()
dataset = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Student Performance")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🎯 Prediction",
            "📊 Analytics",
            "🧠 Feature Importance",
            "📜 Prediction History",
            "ℹ️ About",
        ]
    )

    st.markdown("---")

    if model is not None:
        st.success("✅ Model Loaded")
        st.caption(f"Model: {model_name}")

    else:
        st.error("❌ Model Not Loaded")

    if dataset is not None:
        st.info(
            f"Dataset: {len(dataset):,} students"
        )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎓 Student Performance Prediction System")

    st.markdown(
        """
        ### Machine Learning Based Academic Prediction

        This application uses **15 student attributes** to predict
        expected final examination performance.

        The prediction interface below uses the actual feature names
        instead of generic `Feature 1`, `Feature 2`, etc.
        """
    )

    st.markdown("---")

    if model is None:

        st.error(
            "The prediction model could not be loaded."
        )

        st.warning(
            "Make sure `best_student_model.pkl` and "
            "`preprocessing_elements.pkl` are in the same repository "
            "as this Streamlit application."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Model",
            str(model_name)
        )

        if isinstance(model_info, dict):
            r2 = model_info.get("R2_Score")

            if r2 is not None:
                col2.metric(
                    "R² Score",
                    f"{float(r2):.3f}"
                )
            else:
                col2.metric(
                    "Features",
                    "15 inputs"
                )
        else:
            col2.metric(
                "Features",
                "15 inputs"
            )

        if dataset is not None:

            col3.metric(
                "Students",
                f"{len(dataset):,}"
            )

            if "Final_Exam_Marks" in dataset.columns:

                col4.metric(
                    "Average Final Marks",
                    f"{dataset['Final_Exam_Marks'].mean():.1f}%"
                )

        st.markdown("---")

        st.subheader("📋 Features Used")

        feature_table = pd.DataFrame(
            {
                "No.": range(1, 16),
                "Feature": FEATURES,
                "Type": [
                    "Categorical",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Categorical",
                    "Categorical",
                    "Categorical",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                    "Numeric",
                ]
            }
        )

        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🎯 Prediction":

    st.title("🎯 Student Performance Prediction")

    st.write(
        "Enter all 15 student attributes and click "
        "**Predict Student Performance**."
    )

    if model is None:

        st.error(
            "Model loading failed. Please check the model files and requirements.txt."
        )

        st.stop()

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with st.form("student_prediction_form"):

        st.subheader("👤 Student Information")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col2:

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=100,
                value=20,
                step=1
            )

        with col3:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
                step=0.5
            )

        st.markdown("---")

        st.subheader("📚 Academic Performance")

        col1, col2, col3 = st.columns(3)

        with col1:

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

        with col2:

            assignment_marks = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col3:

            internal_marks = st.number_input(
                "Internal Marks",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            previous_semester_marks = st.number_input(
                "Previous Semester Marks",
                min_value=0.0,
                max_value=100.0,
                value=72.0,
                step=1.0
            )

        with col2:

            project_marks = st.number_input(
                "Project Marks",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col3:

            quiz_marks = st.number_input(
                "Quiz Marks",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0
            )

        st.markdown("---")

        st.subheader("🌐 Lifestyle & Learning Environment")

        col1, col2, col3 = st.columns(3)

        with col1:

            internet_access = st.selectbox(
                "Internet Access",
                ["Yes", "No"]
            )

        with col2:

            family_support = st.selectbox(
                "Family Support",
                ["High", "Medium", "Low"]
            )

        with col3:

            extra_curricular = st.selectbox(
                "Extra-Curricular Activities",
                ["Yes", "No"]
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            sleep_hours = st.number_input(
                "Average Sleep Hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

        with col2:

            daily_screen_time = st.number_input(
                "Daily Screen Time (Hours)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        with col3:

            class_participation = st.number_input(
                "Class Participation",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=1.0
            )

        st.markdown("---")

        predict_button = st.form_submit_button(
            "🚀 Predict Student Performance",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_data = {
            "Gender": gender,
            "Age": int(age),
            "Attendance": float(attendance),
            "Study_Hours": float(study_hours),
            "Assignment_Marks": float(assignment_marks),
            "Internal_Marks": float(internal_marks),
            "Previous_Semester_Marks": float(
                previous_semester_marks
            ),
            "Internet_Access": internet_access,
            "Family_Support": family_support,
            "Extra_Curricular": extra_curricular,
            "Sleep_Hours": float(sleep_hours),
            "Daily_Screen_Time": float(
                daily_screen_time
            ),
            "Project_Marks": float(project_marks),
            "Quiz_Marks": float(quiz_marks),
            "Class_Participation": float(
                class_participation
            ),
        }

        input_df = pd.DataFrame([input_data])

        try:

            with st.spinner(
                "Processing student data and generating prediction..."
            ):

                X_input = prepare_input(
                    input_df,
                    model_features
                )

                prediction = model.predict(X_input)[0]

                prediction = float(
                    np.clip(
                        prediction,
                        0,
                        100
                    )
                )

            category = performance_category(
                prediction
            )

            recommendations = generate_recommendations(
                input_data,
                prediction
            )

            save_prediction(
                input_data,
                prediction,
                category
            )

            st.success(
                "Prediction completed successfully!"
            )

            st.markdown("---")

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.subheader("📊 Prediction Result")

            result1, result2, result3 = st.columns(3)

            result1.metric(
                "Predicted Final Marks",
                f"{prediction:.1f}%"
            )

            result2.metric(
                "Performance Level",
                category
            )

            result3.metric(
                "Model",
                str(model_name)
            )

            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={
                        "text": "Predicted Performance"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": prediction
                        }
                    }
                )
            )

            gauge.update_layout(
                height=350
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            # ------------------------------------------------
            # INPUT CHART
            # ------------------------------------------------

            st.subheader(
                "📈 Student Academic Profile"
            )

            academic_data = pd.DataFrame(
                {
                    "Metric": [
                        "Attendance",
                        "Assignment",
                        "Internal",
                        "Previous Semester",
                        "Project",
                        "Quiz",
                        "Participation",
                    ],
                    "Score": [
                        attendance,
                        assignment_marks,
                        internal_marks,
                        previous_semester_marks,
                        project_marks,
                        quiz_marks,
                        class_participation,
                    ],
                }
            )

            fig = px.bar(
                academic_data,
                x="Metric",
                y="Score",
                range_y=[0, 100],
                title="Student Academic Indicators"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            st.subheader(
                "💡 Recommendations"
            )

            for recommendation in recommendations:

                st.info(
                    "📌 " + recommendation
                )

            # ------------------------------------------------
            # ENTERED VALUES
            # ------------------------------------------------

            with st.expander(
                "🔎 View All 15 Entered Features"
            ):

                display_df = pd.DataFrame(
                    {
                        "Feature": list(
                            input_data.keys()
                        ),
                        "Value": list(
                            input_data.values()
                        ),
                    }
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)

            st.info(
                "This usually means the model and "
                "preprocessing files were created with "
                "different feature structures."
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Student Analytics Dashboard")

    if dataset is None:

        st.warning(
            "Dataset CSV was not found."
        )

        st.stop()

    st.subheader("📌 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students",
        len(dataset)
    )

    if "Final_Exam_Marks" in dataset.columns:

        c2.metric(
            "Average Final Marks",
            f"{dataset['Final_Exam_Marks'].mean():.1f}%"
        )

    if "Attendance" in dataset.columns:

        c3.metric(
            "Average Attendance",
            f"{dataset['Attendance'].mean():.1f}%"
        )

    if "Study_Hours" in dataset.columns:

        c4.metric(
            "Average Study Hours",
            f"{dataset['Study_Hours'].mean():.1f}"
        )

    st.markdown("---")

    if (
        "Attendance" in dataset.columns
        and "Final_Exam_Marks" in dataset.columns
    ):

        st.subheader(
            "📈 Attendance vs Final Exam Marks"
        )

        fig1 = px.scatter(
            dataset,
            x="Attendance",
            y="Final_Exam_Marks",
            trendline="ols",
            title="Attendance and Final Marks"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    with col1:

        if (
            "Study_Hours" in dataset.columns
            and "Final_Exam_Marks" in dataset.columns
        ):

            fig2 = px.scatter(
                dataset,
                x="Study_Hours",
                y="Final_Exam_Marks",
                title="Study Hours vs Final Marks"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    with col2:

        if (
            "Daily_Screen_Time" in dataset.columns
            and "Final_Exam_Marks" in dataset.columns
        ):

            fig3 = px.scatter(
                dataset,
                x="Daily_Screen_Time",
                y="Final_Exam_Marks",
                title="Screen Time vs Final Marks"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

    st.subheader(
        "🔥 Correlation Heatmap"
    )

    numeric_data = dataset.select_dtypes(
        include=np.number
    )

    if not numeric_data.empty:

        correlation = numeric_data.corr()

        fig4 = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "🧠 Feature Importance":

    st.title("🧠 Model Feature Importance")

    if model is None:

        st.error(
            "Model is not loaded."
        )

        st.stop()

    features = model_features

    if features is None:

        if hasattr(
            model,
            "feature_names_in_"
        ):
            features = list(
                model.feature_names_in_
            )

    if features is None:

        st.info(
            "The saved model does not expose feature names."
        )

        st.stop()

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": features,
                "Importance": importance,
            }
        ).sort_values(
            "Importance",
            ascending=True
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            importance_df.sort_values(
                "Importance",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    elif hasattr(model, "coef_"):

        coefficients = np.asarray(
            model.coef_
        ).flatten()

        coef_df = pd.DataFrame(
            {
                "Feature": features,
                "Coefficient": coefficients,
            }
        ).sort_values(
            "Coefficient"
        )

        fig = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Model Coefficients"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Feature importance is not directly available "
            "for this model type."
        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    if not os.path.exists(HISTORY_FILE):

        st.info(
            "No predictions have been recorded yet."
        )

        st.stop()

    try:

        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

        if not history:

            st.info(
                "No prediction history available."
            )

        else:

            rows = []

            for item in history:

                row = {
                    "Time": item.get(
                        "timestamp",
                        ""
                    ),
                    "Predicted Marks": item.get(
                        "predicted_marks",
                        ""
                    ),
                    "Performance": item.get(
                        "performance",
                        ""
                    ),
                }

                rows.append(row)

            history_df = pd.DataFrame(rows)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

            if len(history_df) > 0:

                fig = px.line(
                    history_df,
                    y="Predicted Marks",
                    markers=True,
                    title="Prediction History"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    except Exception as e:

        st.error(
            f"Could not read prediction history: {e}"
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown(
        """
        ## 🎓 Student Performance Prediction System

        This project uses machine learning to estimate a student's
        expected final examination marks from academic, behavioral,
        lifestyle, and learning-environment information.

        ### 15 Input Features

        1. Gender
        2. Age
        3. Attendance
        4. Study Hours
        5. Assignment Marks
        6. Internal Marks
        7. Previous Semester Marks
        8. Internet Access
        9. Family Support
        10. Extra-Curricular Activities
        11. Sleep Hours
        12. Daily Screen Time
        13. Project Marks
        14. Quiz Marks
        15. Class Participation

        ### Technologies

        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Scikit-learn
        - Joblib
        - Plotly

        The model is loaded from the saved `.pkl` file and the
        original preprocessing configuration is reused when available.
        """
    )

    st.success(
        "The application is ready for deployment."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎓 Student Performance Prediction System | "
    "Machine Learning + Streamlit"
)
