import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# Data Preprocessing
df = pd.read_csv(r'c:\Users\BALA\Downloads\Employee-Attrition.csv')
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

df = df.join(pd.get_dummies(df['BusinessTravel'], prefix='Travel')).drop('BusinessTravel', axis=1)
df = df.join(pd.get_dummies(df['Department'], prefix='Dept')).drop('Department', axis=1)
df = df.join(pd.get_dummies(df['EducationField'], prefix='Edu')).drop('EducationField', axis=1)
df = df.join(pd.get_dummies(df['JobRole'], prefix='Job')).drop('JobRole', axis=1)
df = df.join(pd.get_dummies(df['MaritalStatus'], prefix='Status')).drop('MaritalStatus', axis=1)

df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)

st.title("Employee Attrition Analysis and Prediction")
st.markdown("To analyze employee data to visualize Attrition and predict performance.")

# Sidebar preparation

st.sidebar.title("Navigate")
page = st.sidebar.radio("Go to:", ["Dataset Overview", "EDA and its Visualization", "Model Training & Evaluation", "Predict Attrition", "Predict Performance Rating"])

#Dataset Overview

if page == "Dataset Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())
    st.write("**Dataset Shape:**", df.shape)
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

#EDA and its Visualization

elif page == "EDA and its Visualization":
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Attrition', palette='coolwarm', ax=ax)
        ax.set_xticklabels(['Stayed', 'Left'])
        st.pyplot(fig)

    with col2:
        st.subheader("Gender vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Gender', hue='Attrition', palette='coolwarm', ax=ax)
        ax.set_xticklabels(['Female', 'Male'])
        st.pyplot(fig)

    counts = df['PerformanceRating'].value_counts().sort_index()
    count_df = pd.DataFrame({'PerformanceRating': counts.index, 'Rating': counts.values})
    st.subheader("Performance Rating Distribution")
    st.dataframe(count_df, use_container_width=True)
    st.bar_chart(data=count_df, x='PerformanceRating', y='Rating', use_container_width=True)


    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="seismic", center=0, ax=ax)
    st.pyplot(fig)

#Model training & evaluation

elif page == "Model Training & Evaluation":
    st.header("Machine Learning Model Training")
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("Model Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write(f"**Precision:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")
    st.write(f"**AUC-ROC:** {auc:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm',xticklabels=['Predicted: Stayed', 'Predicted: Left'],yticklabels=['Actual: Stayed', 'Actual: Left'],ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Feature Importance")
    feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=feat_imp, y='Feature', x='Importance', palette='viridis', ax=ax)
    st.pyplot(fig)

    #Attrition
elif page == "Predict Attrition":
    st.header("Predict Employee Attrition Probability")

    # Prepare model
    model = RandomForestClassifier(random_state=42)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    st.subheader("Enter Employee Details")

    # User input form
    with st.form("attrition_form"):
        age = st.number_input("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        total_working_years = st.number_input("Total Working Years", 0, 40, 5)
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
        gender = st.selectbox("Gender", ["Female", "Male"])
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_level = st.number_input("Job Level", 1, 5, 2)
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        distance_from_home = st.number_input("Distance From Home (in km)", 1, 50, 5)
        submit = st.form_submit_button("Predict Attrition")

    if submit:
        # Build feature vector (simple subset used for demo)
        input_dict = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'Gender': 1 if gender == 'Male' else 0,
            'OverTime': 1 if overtime == 'Yes' else 0,
            'JobLevel': job_level,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'DistanceFromHome': distance_from_home
        }

        # Align with model columns (fill missing cols with 0)
        input_df = pd.DataFrame([input_dict])
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        st.write(f"**Predicted Probability of Attrition:** {prob:.2%}")
        if prob > 0.5:
            st.error("The employee is likely to leave. Consider retention measures.")
        else:
            st.success("The employee is likely to stay.")


#Predict Performance Rating
elif page == "Predict Performance Rating":
    st.header("Predict Employee Performance Rating")

    # Prepare model
    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)

    st.subheader("Enter Employee Details")

    with st.form("performance_form"):
        age = st.number_input("Age", 18, 60, 30)
        monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
        total_working_years = st.number_input("Total Working Years", 0, 40, 5)
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
        job_level = st.number_input("Job Level", 1, 5, 2)
        training_times_last_year = st.number_input("Training Times Last Year", 0, 10, 3)
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        job_involvement = st.slider("Job Involvement", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        submit = st.form_submit_button("Predict Performance")

    if submit:
        input_dict = {
            'Age': age,
            'MonthlyIncome': monthly_income,
            'TotalWorkingYears': total_working_years,
            'YearsAtCompany': years_at_company,
            'JobLevel': job_level,
            'TrainingTimesLastYear': training_times_last_year,
            'OverTime': 1 if overtime == 'Yes' else 0,
            'JobInvolvement': job_involvement,
            'EnvironmentSatisfaction': environment_satisfaction
        }

        input_df = pd.DataFrame([input_dict])
        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X.columns]

        input_scaled = scaler.transform(input_df)
        predicted_rating = model.predict(input_scaled)[0]

        st.write(f"**Predicted Performance Rating:** {predicted_rating:.2f}")
        if predicted_rating < 3:
            st.error("The employee shows low performance.")
        else:
            st.success("The employee shows good performance.")
