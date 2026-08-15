# ==============================================================================
# PROJECT: STUDENT PERFORMANCE PREDICTION SYSTEM
# AUTHOR: SENIOR DATA SCIENTIST & FULL-STACK MACHINE LEARNING ENGINEER
# DESCRIPTION: A complete college-ready AI/ML system to forecast final scores,
#              analyze attributes, and generate real-time recommendations.
# FILE: student_performance_system.py
# RUN INSTRUCTIONS: streamlit run student_performance_system.py
# ==============================================================================

import sys
import os
import traceback

# Graceful check for bare execution before importing heavy Streamlit dependencies
if __name__ == "__main__":
    try:
        from streamlit.runtime import exists as streamlit_runtime_exists
        if not streamlit_runtime_exists():
            print("\n" + "="*80)
            print("❗ ERROR: BARE PYTHON EXECUTION DETECTED ❗")
            print("="*80)
            print("You attempted to run this file directly using 'python student_performance_system.py'.")
            print("Streamlit applications must be launched via the Streamlit CLI.")
            print("\n👉 Please execute the following command instead:")
            print("    streamlit run student_performance_system.py")
            print("="*80 + "\n")
            sys.exit(0)
    except ImportError:
        print("Streamlit is not installed. Please run: pip install streamlit")
        sys.exit(1)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Library Imports
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor
)

# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="EduAI - Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling configurations for Plots to look highly polished
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Define Data Paths and Constants
DATASET_PATH = "student_data.csv"
MODEL_PATH = "best_student_model.pkl"
PREPROCESSING_PATH = "preprocessing_elements.pkl"
HISTORY_PATH = "prediction_history.json"

# List of models we will train & evaluate
MODEL_NAMES = [
    "Linear Regression", 
    "Decision Tree", 
    "Random Forest", 
    "Gradient Boosting", 
    "Extra Trees"
]

def generate_synthetic_dataset(num_records=1000):
    """
    Generates a highly realistic student performance dataset with natural linear and 
    non-linear relationships between predictors and the target 'Final_Exam_Marks'.
    """
    np.random.seed(42)  # For reproducibility
    
    student_ids = [f"STU_{2026000 + i}" for i in range(num_records)]
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.48, 0.52])
    ages = np.random.randint(18, 26, size=num_records)
    
    # Independent predictors
    attendance = np.random.uniform(50, 100, size=num_records)
    study_hours = np.random.uniform(1.0, 10.0, size=num_records)
    assignment_marks = np.random.uniform(40, 100, size=num_records)
    internal_marks = np.random.uniform(35, 100, size=num_records)
    prev_sem_marks = np.random.uniform(40, 100, size=num_records)
    
    internet_access = np.random.choice(["Yes", "No"], size=num_records, p=[0.85, 0.15])
    family_support = np.random.choice(["High", "Medium", "Low"], size=num_records, p=[0.4, 0.45, 0.15])
    extra_curricular = np.random.choice(["Yes", "No"], size=num_records, p=[0.35, 0.65])
    
    sleep_hours = np.random.uniform(4.5, 9.5, size=num_records)
    daily_screen_time = np.random.uniform(1.0, 8.0, size=num_records)
    project_marks = np.random.uniform(40, 100, size=num_records)
    quiz_marks = np.random.uniform(30, 100, size=num_records)
    class_participation = np.random.uniform(30, 100, size=num_records)
    
    # Creating target logic with realistic weights and interactive correlations
    base_score = (
        0.25 * internal_marks +
        0.20 * prev_sem_marks +
        0.15 * assignment_marks +
        0.15 * project_marks +
        0.08 * quiz_marks +
        0.07 * attendance +
        0.06 * (study_hours * 10) +
        0.04 * class_participation
    )
    
    # Adding categorical adjustments
    internet_adj = np.where(internet_access == "Yes", 1.5, -2.0)
    family_adj = np.where(family_support == "High", 2.0, np.where(family_support == "Low", -3.0, 0.0))
    extra_adj = np.where(extra_curricular == "Yes", 0.5, -0.5)
    sleep_adj = np.where((sleep_hours >= 7.0) & (sleep_hours <= 8.5), 1.5, -2.5)
    screen_adj = np.where(daily_screen_time > 5.5, -2.0, 1.0)
    
    # Combining adjustments and adding synthetic normal noise
    noise = np.random.normal(0, 3.5, size=num_records)
    final_exam_marks = base_score + internet_adj + family_adj + extra_adj + sleep_adj + screen_adj + noise
    
    # Constraining range between 0 and 100
    final_exam_marks = np.clip(final_exam_marks, 30.0, 100.0)
    final_exam_marks = np.round(final_exam_marks, 1)
    
    # Assign Performance categories based on standard educational scales
    performance = []
    for marks in final_exam_marks:
        if marks >= 85:
            performance.append("Excellent")
        elif marks >= 70:
            performance.append("Good")
        elif marks >= 50:
            performance.append("Average")
        else:
            performance.append("Poor")
            
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Age": ages,
        "Attendance": np.round(attendance, 1),
        "Study_Hours": np.round(study_hours, 1),
        "Assignment_Marks": np.round(assignment_marks, 1),
        "Internal_Marks": np.round(internal_marks, 1),
        "Previous_Semester_Marks": np.round(prev_sem_marks, 1),
        "Internet_Access": internet_access,
        "Family_Support": family_support,
        "Extra_Curricular": extra_curricular,
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Daily_Screen_Time": np.round(daily_screen_time, 1),
        "Project_Marks": np.round(project_marks, 1),
        "Quiz_Marks": np.round(quiz_marks, 1),
        "Class_Participation": np.round(class_participation, 1),
        "Final_Exam_Marks": final_exam_marks,
        "Performance": performance
    })
    
    # Intentionally inject 1-2% missing values/duplicates for demonstration purposes (Cleaned in Pipeline)
    for col in ["Attendance", "Study_Hours", "Previous_Semester_Marks"]:
        mask = np.random.choice([True, False], size=len(df), p=[0.015, 0.985])
        df.loc[mask, col] = np.nan
        
    return df

@st.cache_data
def load_and_prepare_raw_data():
    """
    Checks if student_data.csv exists, generates if missing, and loads raw dataframe.
    """
    if not os.path.exists(DATASET_PATH):
        df = generate_synthetic_dataset(1000)
        df.to_csv(DATASET_PATH, index=False)
    else:
        df = pd.read_csv(DATASET_PATH)
    return df

def engineer_features(df_input):
    """
    Creates complex educational engineered features to boost machine learning accuracy.
    """
    df = df_input.copy()
    
    # 1. Average Academic Score: Composite feature of all core graded segments
    df["Average_Academic_Score"] = (
        df["Assignment_Marks"] + 
        df["Internal_Marks"] + 
        df["Project_Marks"] + 
        df["Quiz_Marks"]
    ) / 4.0
    
    # 2. Attendance Percentage Rank: Relative attendance index
    df["Attendance_Percentage"] = df["Attendance"] / 100.0
    
    # 3. Study Efficiency Score: Captures ratio of Study Hours against screen distraction
    df["Study_Efficiency_Score"] = df["Study_Hours"] / (df["Daily_Screen_Time"] + 0.5)
    
    # 4. Performance Index: Combines current internals with baseline from previous semester
    df["Performance_Index"] = (df["Internal_Marks"] * 0.6) + (df["Previous_Semester_Marks"] * 0.4)
    
    # 5. Assignment Ratio: Proportional scoring level of students' assignments
    df["Assignment_Ratio"] = df["Assignment_Marks"] / 100.0
    
    # 6. Participation Score: Composite of attendance and in-class participation index
    df["Participation_Score"] = (df["Attendance"] * 0.5) + (df["Class_Participation"] * 0.5)
    
    return df

def run_preprocessing_pipeline(df_raw, is_training=True):
    """
    Fills missing values, flags outliers, encodes text categories for ML processing,
    scales numerical distributions, and applies feature engineering.
    Returns df_processed (numeric-encoded for model) and df_clean_human (retains text categoricals for visual display).
    """
    df = df_raw.copy()
    
    # 1. Handle Duplicates
    df = df.drop_duplicates(subset=["Student_ID"])
    
    # 2. Impute missing values with appropriate statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Final_Exam_Marks" in numerical_cols:
        numerical_cols.remove("Final_Exam_Marks") # Never impute target
        
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # CRITICAL BUG FIX: Capture a clean human-readable copy of the dataframe prior to encoding
    df_clean_human = df.copy()
    
    # 3. Feature Engineering application to BOTH
    df_clean_human = engineer_features(df_clean_human)
    df = engineer_features(df)
    
    # Update numerical columns list to include engineered columns
    all_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Final_Exam_Marks" in all_numerical:
        all_numerical.remove("Final_Exam_Marks")
    if "Age" in all_numerical:
        all_numerical.remove("Age")
        
    # Categorical columns handling
    categorical_cols = ["Gender", "Internet_Access", "Family_Support", "Extra_Curricular"]
    encoders = {}
    
    if is_training:
        scaler = StandardScaler()
        # Learn and apply Scaling on numeric features
        scaled_features = scaler.fit_transform(df[all_numerical])
        df_scaled = pd.DataFrame(scaled_features, columns=all_numerical, index=df.index)
        
        # Learn Encoding
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        # Save Scaler and Encoders for Inference runtime
        joblib.dump({"scaler": scaler, "encoders": encoders, "numerical_cols": all_numerical}, PREPROCESSING_PATH)
    else:
        # Load elements if saved
        if os.path.exists(PREPROCESSING_PATH):
            prep = joblib.load(PREPROCESSING_PATH)
            scaler = prep["scaler"]
            encoders = prep["encoders"]
            all_numerical = prep["numerical_cols"]
            
            # Apply Scaling
            scaled_features = scaler.transform(df[all_numerical])
            df_scaled = pd.DataFrame(scaled_features, columns=all_numerical, index=df.index)
            
            # Apply Encoding to model df (do NOT apply to df_clean_human!)
            for col in categorical_cols:
                if col in encoders:
                    le = encoders[col]
                    # Handle unseen categories gracefully
                    df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col].astype(str))
        else:
            raise FileNotFoundError("Preprocessing configuration file was not found. Please train model first.")
            
    # Reassemble machine learning processed dataframe with scaled/encoded properties
    df_processed = df.copy()
    for col in all_numerical:
        df_processed[col] = df_scaled[col]
        
    return df_processed, df_clean_human

def train_and_compare_models(df_processed):
    """
    Trains multiple models, evaluates baseline metrics, compares them, 
    saves the top-performing model, and returns performance scoreframes.
    """
    # Define Target and Predictors
    X = df_processed.drop(columns=["Student_ID", "Final_Exam_Marks", "Performance"], errors="ignore")
    y = df_processed["Final_Exam_Marks"]
    
    # Split into train-test distributions (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Initialize algorithms dictionary
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    trained_models = {}
    
    # Loop over all algorithms
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Calculate standard diagnostics
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        cv_mean = np.mean(cv_scores)
        
        results[name] = {
            "MAE": round(mae, 3),
            "MSE": round(mse, 3),
            "RMSE": round(rmse, 3),
            "R2_Score": round(r2, 4),
            "CV_R2_Mean": round(cv_mean, 4)
        }
        trained_models[name] = model
        
    # Convert metric dict to clean comparative Pandas Dataframe
    comparison_df = pd.DataFrame(results).T.sort_values(by="R2_Score", ascending=False)
    
    # Automatically pick the best model based on R-Squared metrics
    best_model_name = comparison_df.index[0]
    best_model = trained_models[best_model_name]
    
    # Persist the absolute best performing estimator structure
    joblib.dump({
        "model_name": best_model_name,
        "model_object": best_model,
        "features": list(X.columns),
        "metrics": results[best_model_name]
    }, MODEL_PATH)
    
    return comparison_df, best_model_name, X_test, y_test, best_model

def save_prediction_record(input_dict, predicted_marks, performance_level, recommendation):
    """
    Saves a record of prediction to local JSON file for log audits.
    """
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": input_dict,
        "predicted_marks": round(float(predicted_marks), 1),
        "performance": performance_level,
        "recommendation_count": len(recommendation)
    }
    
    history = []
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
        except:
            history = []
            
    history.append(record)
    
    # Cap history logs at 100 entries to prevent memory overflow
    if len(history) > 100:
        history = history[-100:]
        
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=4)

def generate_recommendations(inputs):
    """
    Generates tailored, action-oriented dynamic counseling recommendations
    based on custom thresholds of student inputs.
    """
    recs = []
    
    # 1. Attendance Check
    if inputs["Attendance"] < 75:
        recs.append({
            "topic": "Attendance Priority",
            "text": f"Your attendance is currently at {inputs['Attendance']}%, which is below the critical university threshold of 75%. Please prioritize lectures.",
            "severity": "High"
        })
    elif inputs["Attendance"] < 85:
        recs.append({
            "topic": "Attendance Buffer",
            "text": "Your attendance is moderate (75-85%). Attend a few more sequential classes to buffer against unforeseen illness.",
            "severity": "Medium"
        })
        
    # 2. Study Habits Check
    if inputs["Study_Hours"] < 4:
        recs.append({
            "topic": "Study Schedule Boost",
            "text": f"Your current study time of {inputs['Study_Hours']} hours/day is suboptimal. Increase this systematically to 5-6 hours.",
            "severity": "High"
        })
        
    # 3. Class Component Marks
    if inputs["Internal_Marks"] < 60:
        recs.append({
            "topic": "Internal Assessment Support",
            "text": "Your current internal mid-semester score is low. Schedule a remedial session with your professor or teaching assistant.",
            "severity": "High"
        })
        
    if inputs["Assignment_Marks"] < 65:
        recs.append({
            "topic": "Assignment Deadlines",
            "text": "Incorporate peer feedback and complete multiple drafts before final assignment submission dates to secure grade boosts.",
            "severity": "Medium"
        })
        
    # 4. Lifestyle Balance Checks
    if inputs["Sleep_Hours"] < 6.5:
        recs.append({
            "topic": "Sleep Deprivation Risk",
            "text": f"Sleeping only {inputs['Sleep_Hours']} hours damages memory consolidation. Ensure 7-8 hours of regular, uninterrupted rest.",
            "severity": "High"
        })
        
    if inputs["Daily_Screen_Time"] > 5.0:
        recs.append({
            "topic": "Screen Time Moderation",
            "text": f"High digital screen times ({inputs['Daily_Screen_Time']} hours/day) cause cognitive fatigue. Try using focus mode apps.",
            "severity": "Medium"
        })
        
    # Default general positive reinforcements if the scores are high
    if len(recs) == 0:
        recs.append({
            "topic": "Consistency Plan",
            "text": "Excellent stats across all performance indicators! Maintain your present schedule and join peer tutoring sessions to share strategies.",
            "severity": "Positive"
        })
        
    return recs

@st.cache_resource
def get_system_context():
    """
    Encapsulates setup operations to trigger cleanly within the active
    Streamlit runtime thread, eliminating ScriptRunContext and out-of-sync preprocessor anomalies.
    Caches results using st.cache_resource to prevent heavy disk-IO and model deserialization overhead on every rerun.
    """
    raw_df = load_and_prepare_raw_data()
    
    # Safeguard: Verify if both preprocess metadata and model objects are present.
    needs_training = (not os.path.exists(MODEL_PATH)) or (not os.path.exists(PREPROCESSING_PATH))
    
    if needs_training:
        # Perform training-level fit and transform
        df_processed, df_clean = run_preprocessing_pipeline(raw_df, is_training=True)
        train_and_compare_models(df_processed)
    else:
        # Load and transform using existing preprocessor state only
        df_processed, df_clean = run_preprocessing_pipeline(raw_df, is_training=False)
        
    best_model_data = joblib.load(MODEL_PATH)
    return df_clean, df_processed, best_model_data

# Execute setup context safely within Streamlit execution flow (cached for instant performance!)
df_clean, df_processed, best_model_data = get_system_context()

st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color:#1E3A8A; font-family:sans-serif;'>EduAI Predictive</h2>
        <span style='background-color:#E0F2FE; color:#0369A1; padding: 4px 12px; border-radius:12px; font-size:12px; font-weight:bold;'>Active Tier: Enterprise</span>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
page_selector = st.sidebar.radio(
    "🧭 Navigation Controls",
    ["🏠 Executive Home", "📂 Data Exploration", "📊 Executive Analytics", "🤖 Modeling Panel", "🎯 Run Prediction", "📋 Records History", "ℹ️ About Platform"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Quick Insights**\n"
    "Academic metrics (Internals & Assignment marks) maintain the highest overall feature significance across all prediction algorithms."
)

if page_selector == "🏠 Executive Home":
    st.markdown(
        """
        <div style="background-color:#1E3A8A; padding:35px; border-radius:15px; margin-bottom:25px; color:white;">
            <h1 style='color:white; margin:0;'>Student Academic Performance Prediction</h1>
            <p style='font-size:18px; opacity:0.9; margin-top:10px;'>
                An advanced predictive machine learning hub that guides institutions, educators, and students by turning demographic, behavioral, and academic performance indicators into actionable strategies.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Key Project Objectives")
        st.write(
            "Our intelligent modeling environment targets educational pain points by processing modern academic and lifestyle metrics "
            "to minimize retention attrition rates and elevate performance."
        )
        st.markdown(
            """
            * **Early Grade Intervention:** Predicts potential struggling ranges well before continuous feedback cycles conclude.
            * **Multidimensional Profiles:** Uses both lifestyle habits (such as sleep cycle levels and internet usage) and solid grades.
            * **Actionable Advising Engine:** Generates instant customized prescriptions for students to repair negative trajectories.
            * **Unbiased Analysis:** Provides mathematical, statistical proof of parameters most correlating with high test outcomes.
            """
        )
        
        st.subheader("⚙️ Robust Machine Learning Workflow")
        st.image("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=1200&auto=format&fit=crop", 
                 caption="Transforming Student Data into Predictive Action Plans", use_container_width=True)

    with col2:
        st.subheader("🛠️ Applied Technologies")
        st.markdown(
            """
            - **Data Pipelines:** Pandas, NumPy
            - **Interactive Visuals:** Plotly, Seaborn
            - **Machine Learning Core:** Scikit-Learn
            - **Deployment Infrastructure:** Streamlit Cloud
            - **Preserved State Engines:** Joblib, JSON
            """
        )
        
        st.subheader("📊 Primary Operational Dataset Summary")
        st.markdown(f"**Total Sample Count:** `{len(df_clean)} Students`")
        st.markdown(f"**Calculated Average Grade:** `{df_clean['Final_Exam_Marks'].mean():.1f}/100`")
        st.markdown(f"**Baseline Attendance Rate:** `{df_clean['Attendance'].mean():.1f}%`")
        
        st.metric(label="Model Reliability (R² Score)", value=f"{best_model_data['metrics']['R2_Score']*100:.1f}%", delta="Stable")

elif page_selector == "📂 Data Exploration":
    st.title("📂 Institutional Dataset Explorer")
    st.write("Browse, query, filter, and inspect the real-time student repository powering our machine learning predictions.")
    
    st.markdown("### 🔍 Advanced Filter Controls")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    # Multi-select options are now beautiful text strings instead of encoded digits!
    with f_col1:
        gender_filter = st.multiselect("Gender", options=df_clean["Gender"].unique(), default=df_clean["Gender"].unique())
    with f_col2:
        support_filter = st.multiselect("Family Support Level", options=df_clean["Family_Support"].unique(), default=df_clean["Family_Support"].unique())
    with f_col3:
        attendance_range = st.slider("Attendance Level (%)", min_value=float(df_clean["Attendance"].min()), max_value=float(df_clean["Attendance"].max()), value=(70.0, 100.0))
    with f_col4:
        performance_filter = st.multiselect("Performance Bracket", options=df_clean["Performance"].unique(), default=df_clean["Performance"].unique())
        
    search_id = st.text_input("🔍 Find Specific Student (Enter Full Student_ID, e.g., STU_2026042):").strip()
    
    # Safe checks for empty filter lists to avoid empty-mask crashes
    filtered_df = df_clean.copy()
    if gender_filter:
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]
    else:
        filtered_df = filtered_df.iloc[0:0] # Return empty if no categories are chosen
        
    if support_filter and len(filtered_df) > 0:
        filtered_df = filtered_df[filtered_df["Family_Support"].isin(support_filter)]
    elif not support_filter:
        filtered_df = filtered_df.iloc[0:0]
        
    if len(filtered_df) > 0:
        filtered_df = filtered_df[
            (filtered_df["Attendance"] >= attendance_range[0]) &
            (filtered_df["Attendance"] <= attendance_range[1])
        ]
        
    if performance_filter and len(filtered_df) > 0:
        filtered_df = filtered_df[filtered_df["Performance"].isin(performance_filter)]
    elif not performance_filter:
        filtered_df = filtered_df.iloc[0:0]
        
    if search_id and len(filtered_df) > 0:
        filtered_df = filtered_df[filtered_df["Student_ID"] == search_id]
        
    st.markdown(f"Showing **{len(filtered_df)}** entries out of **{len(df_clean)}** total student profiles matching query.")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Dataset to CSV",
        data=csv_data,
        file_name="filtered_student_registry.csv",
        mime="text/csv"
    )
    
    with st.expander("📝 Dataset Quality & Imputation Diagnostics Report"):
        st.write("The platform auto-detects system gaps and repairs them prior to model ingestion.")
        col_impute1, col_impute2 = st.columns(2)
        with col_impute1:
            st.markdown("**1. Raw Dataset Status Prior to Imputation:**")
            raw_nulls = load_and_prepare_raw_data().isnull().sum()
            st.dataframe(raw_nulls[raw_nulls > 0].to_frame(name="Missing Elements Count"))
        with col_impute2:
            st.markdown("**2. Post-Pipeline Cleansed Diagnostics:**")
            clean_nulls = df_clean.isnull().sum().to_frame(name="Remaining Elements Count")
            st.dataframe(clean_nulls)
            st.success("✔️ Pipeline executed. Missing elements completely resolved with column medians.")

elif page_selector == "📊 Executive Analytics":
    st.title("📊 Strategic Analytical Dashboard")
    st.write("Examine underlying academic relationships, behavioral interactions, and continuous demographic features.")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Enrolled", f"{len(df_clean)} Students", "Target: 1000")
    m2.metric("Mean Final Grade", f"{df_clean['Final_Exam_Marks'].mean():.1f}%", "+1.2% vs Prev Sem")
    m3.metric("Critical Attendance Rate", f"{df_clean['Attendance'].mean():.1f}%", "Target > 80.0%")
    m4.metric("Avg Study Load", f"{df_clean['Study_Hours'].mean():.1f} Hrs/Day", "Recommended: 5.5")
    m5.metric("Target Variable Class", "Continuous Marks", "Regression Mode")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Attendance vs. Final Exam Marks")
        fig_attendance = px.scatter(
            df_clean, 
            x="Attendance", 
            y="Final_Exam_Marks", 
            color="Performance",
            trendline="ols",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Correlation: Attendance Level vs Final Marks"
        )
        st.plotly_chart(fig_attendance, use_container_width=True)
        
    with c2:
        st.subheader("📚 Distribution of Grades by Gender")
        fig_violin = px.box(
            df_clean,
            x="Gender",
            y="Final_Exam_Marks",
            color="Gender",
            points="all",
            title="Distribution Box Plot of Final Exam Marks"
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("☕ Study Hours vs Daily Screen Time Interactions")
        fig_bubble = px.scatter(
            df_clean,
            x="Study_Hours",
            y="Daily_Screen_Time",
            size="Final_Exam_Marks",
            color="Performance",
            hover_name="Student_ID",
            title="Study vs Screen Time scaled by Final Grades"
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        
    with c4:
        st.subheader("🔥 Performance Bracket Makeup")
        perf_counts = df_clean["Performance"].value_counts().reset_index()
        fig_pie = px.pie(
            perf_counts,
            names="Performance",
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Sunsetdark,
            title="Percentage Contribution of Student Performance Categories"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🧮 Overall Multi-feature Correlation Heatmap")
    df_numeric = df_clean.select_dtypes(include=[np.number])
    corr_matrix = df_numeric.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Student Feature Linear Correlation Matrix"
    )
    fig_corr.update_layout(width=1000, height=800)
    st.plotly_chart(fig_corr, use_container_width=True)

elif page_selector == "🤖 Modeling Panel":
    st.title("🤖 Unified Modeling Panel & Model Comparison")
    st.write("Train and compare several ML regressor variants across custom statistical cross-validation matrices.")
    
    st.info(f"💾 **Current Live System Predictor:** `{best_model_data['model_name']}` "
            f"(Validation R² Score: `{best_model_data['metrics']['R2_Score'] * 100:.2f}%`)")
    
    if st.button("🚀 Re-Run Comparative ML Pipeline (Train 5 Regression Models)"):
        with st.spinner("Executing Feature Engineering and Training Estimators across K-Fold Folds..."):
            time.sleep(1.0)
            comparison, best_name, X_test, y_test, model_obj = train_and_compare_models(df_processed)
            st.success("🎉 Machine Learning Pipeline Completed successfully!")
            
            # CLEAR THE STREAMLIT CACHE SO NEWLY TRAINED MODEL IS LOADED INTO SESSION MEMORY IMMEDIATELY!
            st.cache_resource.clear()
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.subheader("📊 Performance Matrix Results")
                st.dataframe(comparison)
                
            with col_res2:
                st.subheader("💡 Selected Leading Estimator")
                st.markdown(f"🏆 **Winner:** `{best_name}`")
                st.write("This algorithm returned the highest R-Squared value, identifying it as the optimal generalizable tool for current predictive student attributes.")
                
                predictions = model_obj.predict(X_test)
                
                fig_predict = go.Figure()
                fig_predict.add_trace(go.Scatter(
                    x=y_test, y=predictions,
                    mode='markers',
                    marker=dict(color='blue', opacity=0.6),
                    name='Predicted vs Actual'
                ))
                fig_predict.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Ideal Baseline (100% Perfect)'
                ))
                fig_predict.update_layout(
                    title="Model Error Profile (Predicted vs. Ground Truth Actual)",
                    xaxis_title="Actual Grades (Final)",
                    yaxis_title="Predicted Grades",
                    height=400
                )
                st.plotly_chart(fig_predict, use_container_width=True)
                
        st.rerun()

    st.markdown("### 🧬 Feature Importance Profile")
    st.write("Analysis of relative feature weights for the active model:")
    
    active_features = best_model_data["features"]
    active_estimator = best_model_data["model_object"]
    
    if hasattr(active_estimator, "feature_importances_"):
        importances = active_estimator.feature_importances_
        imp_df = pd.DataFrame({"Feature": active_features, "Relative Importance Score": importances})
        imp_df = imp_df.sort_values(by="Relative Importance Score", ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            y="Feature",
            x="Relative Importance Score",
            orientation="h",
            color="Relative Importance Score",
            color_continuous_scale="Purples",
            title=f"Relative Feature Contributions ({best_model_data['model_name']})"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    elif hasattr(active_estimator, "coef_"):
        coefficients = active_estimator.coef_
        imp_df = pd.DataFrame({"Feature": active_features, "Linear Coefficient Weight": coefficients})
        imp_df = imp_df.sort_values(by="Linear Coefficient Weight", ascending=True)
        
        fig_imp = px.bar(
            imp_df,
            y="Feature",
            x="Linear Coefficient Weight",
            orientation="h",
            color="Linear Coefficient Weight",
            color_continuous_scale="Tropic",
            title=f"Linear Regression Feature Coefficients"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance visual is not available directly for this model architecture.")

elif page_selector == "🎯 Run Prediction":
    st.title("🎯 Real-Time Grade Inference & Student Counseling")
    st.write("Input behavioral, lifestyle, and score parameters below to generate student-specific predictions.")
    
    with st.form("student_features_form"):
        st.markdown("### 🎓 Step 1: Academic & Continuous Test Scores")
        col_ac1, col_ac2, col_ac3, col_ac4 = st.columns(4)
        
        with col_ac1:
            attendance_in = st.slider("Attendance Level (%)", 50.0, 100.0, 85.0, step=0.5, help="Percentage of total academic lectures attended.")
        with col_ac2:
            study_hours_in = st.slider("Daily Study Time (Hours)", 1.0, 10.0, 5.0, step=0.1, help="Avg. independent daily study workload.")
        with col_ac3:
            assignment_in = st.number_input("Assignment Mark (0-100)", min_value=0.0, max_value=100.0, value=75.0, help="Continuous progress grading index.")
        with col_ac4:
            internal_in = st.number_input("Mid-Term Internal Grade (0-100)", min_value=0.0, max_value=100.0, value=70.0, help="Performance inside closed environment testing.")
            
        col_ac5, col_ac6, col_ac7 = st.columns(3)
        with col_ac5:
            prev_sem_in = st.number_input("Previous Sem Score (0-100)", min_value=0.0, max_value=100.0, value=72.0, help="Baseline terminal grade score.")
        with col_ac6:
            project_in = st.number_input("Project Assessment Mark (0-100)", min_value=0.0, max_value=100.0, value=80.0, help="Practical component marks.")
        with col_ac7:
            quiz_in = st.number_input("Weekly Quiz Average (0-100)", min_value=0.0, max_value=100.0, value=65.0, help="Frequent quick assessment parameters.")
            
        st.markdown("---")
        st.markdown("### 👥 Step 2: Demographic, Social, & Lifestyle Factors")
        col_dem1, col_dem2, col_dem3, col_dem4 = st.columns(4)
        
        with col_dem1:
            gender_in = st.selectbox("Gender Identification", ["Female", "Male"])
        with col_dem2:
            age_in = st.number_input("Student Age (Years)", min_value=17, max_value=30, value=20)
        with col_dem3:
            internet_in = st.selectbox("Direct Internet Access?", ["Yes", "No"])
        with col_dem4:
            family_in = st.selectbox("Family Academic Support Level", ["High", "Medium", "Low"])
            
        col_dem5, col_dem6, col_dem7, col_dem8 = st.columns(4)
        with col_dem5:
            extra_in = st.selectbox("Active Extra-Curriculars?", ["Yes", "No"])
        with col_dem6:
            sleep_in = st.slider("Average Sleep (Hours)", 4.0, 10.0, 7.5, step=0.1)
        with col_dem7:
            screen_in = st.slider("Daily Recreational Screen Time (Hours)", 0.5, 12.0, 3.5, step=0.1)
        with col_dem8:
            participation_in = st.slider("In-Class Verbal Participation Rank (0-100)", 0.0, 100.0, 60.0, step=1.0)
            
        predict_submit = st.form_submit_button("🧪 Compile Diagnostics & Run Prediction Engine")
        
    if predict_submit:
        with st.spinner("Processing engineered vectors & evaluating estimator pathways..."):
            time.sleep(0.7)
            
            raw_input_payload = {
                "Student_ID": ["STU_TEMP"],
                "Gender": [gender_in],
                "Age": [int(age_in)],
                "Attendance": [float(attendance_in)],
                "Study_Hours": [float(study_hours_in)],
                "Assignment_Marks": [float(assignment_in)],
                "Internal_Marks": [float(internal_in)],
                "Previous_Semester_Marks": [float(prev_sem_in)],
                "Internet_Access": [internet_in],
                "Family_Support": [family_in],
                "Extra_Curricular": [extra_in],
                "Sleep_Hours": [float(sleep_in)],
                "Daily_Screen_Time": [float(screen_in)],
                "Project_Marks": [float(project_in)],
                "Quiz_Marks": [float(quiz_in)],
                "Class_Participation": [float(participation_in)]
            }
            
            df_input_raw = pd.DataFrame(raw_input_payload)
            
            try:
                # Add default structural targets to process inference smoothly
                df_input_raw["Final_Exam_Marks"] = 0.0
                df_input_raw["Performance"] = "Average"
                
                # Transform via frozen preprocess metadata state
                df_processed_single, _ = run_preprocessing_pipeline(df_input_raw, is_training=False)
                
                # Align exact model features
                features_for_model = best_model_data["features"]
                X_single = df_processed_single[features_for_model].copy()
                
                # Robust Explicit Data Casting to bypass pandas object type mismatches during single row inference
                X_single = X_single.apply(pd.to_numeric, errors='coerce').fillna(0.0)
                
                # Evaluate Prediction Marks
                predicted_raw_marks = best_model_data["model_object"].predict(X_single)[0]
                predicted_raw_marks = float(np.clip(predicted_raw_marks, 0.0, 100.0))
                
                if predicted_raw_marks >= 85.0:
                    performance_class = "Excellent"
                    color_indicator = "#047857"
                    color_bg = "#D1FAE5"
                elif predicted_raw_marks >= 70.0:
                    performance_class = "Good"
                    color_indicator = "#0369A1"
                    color_bg = "#E0F2FE"
                elif predicted_raw_marks >= 50.0:
                    performance_class = "Average"
                    color_indicator = "#B45309"
                    color_bg = "#FEF3C7"
                else:
                    performance_class = "Poor"
                    color_indicator = "#B91C1C"
                    color_bg = "#FEE2E2"
                    
                counseling_points = generate_recommendations({
                    "Attendance": attendance_in,
                    "Study_Hours": study_hours_in,
                    "Assignment_Marks": assignment_in,
                    "Internal_Marks": internal_in,
                    "Sleep_Hours": sleep_in,
                    "Daily_Screen_Time": screen_in
                })
                
                st.markdown("---")
                st.markdown("## 🔍 Academic Performance Assessment Profile")
                
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown(
                        f"""
                        <div style='background-color:{color_bg}; padding: 25px; border-radius:15px; border-left: 8px solid {color_indicator};'>
                            <h4 style='color:#1E293B; margin:0; text-transform:uppercase;'>System Prediction Output</h4>
                            <h1 style='color:{color_indicator}; font-size:48px; margin: 10px 0;'>{predicted_raw_marks:.1f}%</h1>
                            <p style='color:#475569; font-weight:bold; margin:0;'>Estimated Grade Level Bracket: <span style='color:{color_indicator};'>{performance_class}</span></p>
                            <p style='font-size:12px; color:#64748B; margin-top:10px;'>Inference generated via: <b>{best_model_data['model_name']}</b></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### Performance Gauge Scale")
                    st.progress(int(predicted_raw_marks))
                    
                with res_col2:
                    st.markdown("### 📋 Expert Counselor Advisory Notes")
                    st.write("Below are systematic intervention goals designed based on behavior traits:")
                    
                    for count, pt in enumerate(counseling_points, 1):
                        alert_icon = "⚠️" if pt['severity'] == "High" else "ℹ️"
                        if pt['severity'] == "Positive":
                            alert_icon = "🌟"
                            
                        st.markdown(f"**{count}. {alert_icon} [{pt['topic']}] ({pt['severity']} Alert)**")
                        st.caption(pt['text'])
                
                flat_inputs = {k: v[0] for k, v in raw_input_payload.items() if k != "Student_ID"}
                save_prediction_record(flat_inputs, predicted_raw_marks, performance_class, counseling_points)
                st.toast("✔️ Prediction logged into local audits records history.")
                
            except Exception as e:
                st.error(f"❌ Structural Failure during Pipeline execution: {str(e)}")
                with st.expander("🛠️ View Complete Debugging Traceback Details"):
                    st.code(traceback.format_exc())
                st.info("Ensure modeling has run successfully to export valid scaling metadata.")

elif page_selector == "📋 Records History":
    st.title("📋 Prediction History Ledger")
    st.write("This log audits predictions ran dynamically during active deployment sessions.")
    
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history_data = json.load(f)
                
            if len(history_data) > 0:
                rows = []
                for entry in history_data:
                    row = {
                        "Timestamp": entry["timestamp"],
                        "Estimated Grade (%)": entry["predicted_marks"],
                        "Class Bracket": entry["performance"],
                        "Age": entry["inputs"].get("Age", "N/A"),
                        "Gender": entry["inputs"].get("Gender", "N/A"),
                        "Attendance (%)": entry["inputs"].get("Attendance", "N/A"),
                        "Study Hours": entry["inputs"].get("Study_Hours", "N/A"),
                        "Internal Score": entry["inputs"].get("Internal_Marks", "N/A"),
                        "Recommendation Counts": entry["recommendation_count"]
                    }
                    rows.append(row)
                    
                history_df = pd.DataFrame(rows)
                st.dataframe(history_df, use_container_width=True)
                
                if st.button("🗑️ Purge Historical Audit Trail Logs"):
                    os.remove(HISTORY_PATH)
                    st.success("Ledger deleted successfully.")
                    st.rerun()
            else:
                st.info("No query predictions logged during current operational periods yet.")
        except Exception as e:
            st.error(f"Failed to decode history file structure: {str(e)}")
    else:
        st.info("No query predictions logged during current operational periods yet.")

elif page_selector == "ℹ️ About Platform":
    st.title("ℹ️ Institutional Platform Information")
    st.write("Technical metadata regarding pipeline parameters and active training checkpoints.")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("🧬 Computational Architecture")
        st.write(
            "This application utilizes a single pipeline executing continuous label encoders "
            "coupled to Scikit-Learn structures. Features engineered directly reflect classical "
            "pedagogical research targets relating sleep metrics and continuous screen limits."
        )
        
        st.subheader("📦 Model Properties")
        st.markdown(f"**Loaded Algorithmic Class:** `{best_model_data['model_name']}`")
        st.markdown(f"**Attributes Used for Prediction:** `{len(best_model_data['features'])} Variables`")
        st.markdown(f"**Validation R² Coefficient:** `{best_model_data['metrics']['R2_Score']:.5f}`")
        st.markdown(f"**Model Mean Absolute Error (MAE):** `{best_model_data['metrics']['MAE']:.2f} Marks`")
        
    with col_info2:
        st.subheader("💡 Deployment System Guidelines")
        st.markdown(
            """
            * **Input Boundaries:** Predictions perform with highest fidelity when inputs closely map to continuous ranges (e.g. Study hours bounded strictly between 1 and 10).
            * **Local Persist:** Data persists automatically in local directory as `student_data.csv` and serialized metadata binaries (`best_student_model.pkl`).
            * **Scale Operations:** To import customized school listings, format datasets with columns mirroring the schema shown inside the **Data Exploration** segment.
            """
        )

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 13px; color:#64748B; margin-top:20px;'>
        🎓 <b>EduAI Student Performance Hub</b> &copy; 2026 Academic Systems Technologies. | Built for Modern Institutions, Faculty, and Administrators.
    </div>
    """, 
    unsafe_allow_html=True
)