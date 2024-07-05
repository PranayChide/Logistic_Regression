# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:44:40 2024

@author: admin
"""

import streamlit as st 
import pandas as pd
import numpy as np
import joblib


st.title("Logistic Regression Model")

# Load the trained model and preprocessor
model = joblib.load('logistic_regression.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Define the Streamlit app
def main():
    st.title("Titanic Survival Prediction")
    
    st.markdown("""
    Enter the passenger details to predict the survival probability.
    """)

    # Define user inputs
    passengerID = st.selectbox("PassengerId",[1,2,3])
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    name = st.selectbox("Name",["Braund, Mr. Owen Harris","Cumings, Mrs. John Bradley (Florence Briggs Th...","Heikkinen, Miss. Laina"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.slider("Parents/Children Aboard", 0, 8, 0)
    ticket = st.selectbox("Ticket",["A/5 21171","PC 17599","STON/O2. 3101282"])
    fare = st.slider("Fare", 0, 500, 50)
    cabin = st.selectbox("Cabin",["NaN","C85","NaN"])
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    
    
    # Create a dictionary with the input data
    input_data = {
        "PassengerId": passengerID,
        "Pclass": pclass,
        "Sex": sex,
        "Name": name,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin,
        "Embarked": embarked
    }
    
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    input_preprocessed = preprocessor.transform(input_df)
    
    # Make a prediction
    prediction = model.predict(input_preprocessed)
    prediction_proba = model.predict_proba(input_preprocessed)[:, 1]
    
    # Display the prediction
    if prediction[0] == 1:
        st.success(f"The passenger is likely to survive with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.error(f"The passenger is unlikely to survive with a probability of {prediction_proba[0]:.2f}.")
        
if __name__ == "__main__":
    main()




