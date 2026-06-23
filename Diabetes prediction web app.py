import numpy as np
import pandas as pd
import pickle
import streamlit as st
import os

# Get current directory (works both locally and online)
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "trained_model.sav")
scaler_path = os.path.join(current_dir, "scaler.sav")

# Load the saved model and scaler
with open(model_path, "rb") as file:
    loaded_model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)


# Function for prediction
def diabetes_prediction(input_data):
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    try:
        input_data = pd.DataFrame([list(map(float, input_data))], columns=feature_names)
    except ValueError:
        return "⚠️ Please enter valid numerical values for all fields."

    input_data = scaler.transform(input_data)
    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


def main():
    st.title("🩺 Diabetes Prediction Web App")
    st.markdown("Provide the patient's details below to predict the likelihood of diabetes.")
    st.warning("⚠️ Disclaimer: This app is for educational purposes only and not a substitute for professional medical advice.")
    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        SkinThickness = st.text_input("Skin Thickness value")
        BMI = st.text_input("BMI value")

    with col2:
        Glucose = st.text_input("Glucose Level")
        Insulin = st.text_input("Insulin Level")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")

    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
        Age = st.text_input("Age of the Person")

    st.write("---")

    if st.button("🔍 Get Diabetes Test Result"):
        if "" in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
            st.warning("⚠️ Please fill in all fields before testing.")
        else:
            diagnosis = diabetes_prediction([
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ])
            st.success(diagnosis)


if __name__ == "__main__":
    main()