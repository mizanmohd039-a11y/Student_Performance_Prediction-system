import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("🎓 Student Performance Prediction System")
st.write("Predict student performance using a Machine Learning model.")

st.divider()

# ---------------------------------------------------------
# FIND MODEL FILE
# ---------------------------------------------------------
MODEL_FILES = [
    "best_student_model.pkl",
    "student_performance_model.pkl",
    "best_student_model.joblib",
    "student_performance_model.joblib"
]

model = None
model_file = None

for file in MODEL_FILES:
    if os.path.exists(file):
        try:
            model = joblib.load(file)
            model_file = file
            break
        except Exception:
            pass

if model is None:
    st.error(
        "❌ Model file was not found. "
        "Make sure your .pkl model file is uploaded to the GitHub repository."
    )
    st.stop()

st.success(f"✅ Model loaded successfully: `{model_file}`")

# ---------------------------------------------------------
# GET MODEL FEATURES
# ---------------------------------------------------------
features = None

if hasattr(model, "feature_names_in_"):
    features = list(model.feature_names_in_)

elif hasattr(model, "named_steps"):
    for step_name, step in model.named_steps.items():
        if hasattr(step, "feature_names_in_"):
            features = list(step.feature_names_in_)
            break

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🎓 Student Information")

st.sidebar.info(
    "Enter the student's information below and click "
    "**Predict Performance**."
)

# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------
st.subheader("📋 Enter Student Details")

# If the model contains feature names, create inputs automatically.
if features:

    st.write("Model features detected:")

    user_values = {}

    columns = st.columns(2)

    for i, feature in enumerate(features):

        with columns[i % 2]:

            feature_name = str(feature)
            lower_name = feature_name.lower()

            # Common categorical fields
            if any(word in lower_name for word in [
                "gender",
                "sex"
            ]):
                user_values[feature] = st.selectbox(
                    feature_name,
                    ["Male", "Female"]
                )

            elif any(word in lower_name for word in [
                "yes", "no",
                "internet",
                "activity",
                "support",
                "school",
                "family"
            ]):
                user_values[feature] = st.selectbox(
                    feature_name,
                    ["Yes", "No"]
                )

            else:
                user_values[feature] = st.number_input(
                    feature_name,
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0
                )

else:

    st.warning(
        "⚠️ The model does not expose feature names. "
        "Enter the five features used by your model."
    )

    col1, col2 = st.columns(2)

    with col1:
        feature_1 = st.number_input(
            "Feature 1",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        feature_2 = st.number_input(
            "Feature 2",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        feature_3 = st.number_input(
            "Feature 3",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

    with col2:
        feature_4 = st.number_input(
            "Feature 4",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

        feature_5 = st.number_input(
            "Feature 5",
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

    user_values = {
        "Feature 1": feature_1,
        "Feature 2": feature_2,
        "Feature 3": feature_3,
        "Feature 4": feature_4,
        "Feature 5": feature_5
    }

# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
st.divider()

if st.button(
    "🔮 Predict Student Performance",
    type="primary",
    use_container_width=True
):

    try:

        # Create DataFrame
        input_data = pd.DataFrame([user_values])

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------
        prediction = model.predict(input_data)[0]

        st.subheader("📊 Prediction Result")

        # Display prediction
        st.success(
            f"🎯 Predicted Performance: **{prediction}**"
        )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            st.subheader("📈 Prediction Probability")

            classes = getattr(
                model,
                "classes_",
                range(len(probabilities))
            )

            probability_data = pd.DataFrame({
                "Class": classes,
                "Probability": probabilities
            })

            probability_data["Probability"] = (
                probability_data["Probability"] * 100
            ).round(2)

            st.dataframe(
                probability_data,
                use_container_width=True
            )

            st.bar_chart(
                probability_data.set_index("Class")["Probability"]
            )

        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------
        st.subheader("📝 Student Information")

        display_data = pd.DataFrame(
            list(user_values.items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.write(
            "This usually happens when the input features "
            "do not match the features used when training the model."
        )

        st.code(str(e))

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Machine Learning + Streamlit"
)
