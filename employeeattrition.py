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

# Sidebar Navigation 

st.sidebar.title("Navigation")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'], ax=ax)
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

    model = RandomForestClassifier(random_state=42)
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    employee = st.selectbox("Select Employee (by index):", df.index)
    input_data = df.loc[[employee]].drop('Attrition', axis=1)
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"**Predicted Probability of Attrition:** {prob:.2%}")

    if prob > 0.5:
        st.error("The employee is likely to leave, consider beginning proactive retention strategies.")
    else:
        st.success("The employee is likely to stay.")

#Predict Performance rating

elif page == "Predict Performance Rating":
    st.header("Predict Employee Performance Rating")

    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)

    employee = st.selectbox("Select Employee (by index):", df.index)
    input_data = df.loc[[employee]].drop('PerformanceRating', axis=1)
    input_scaled = scaler.transform(input_data)

    predicted_rating = model.predict(input_scaled)[0]

    st.write(f"**Predicted Performance Rating:** {predicted_rating:.2f}")

    if predicted_rating < 3:
        st.error("The employee has low performance.")
    else:
        st.success("The employee has good performance.")
