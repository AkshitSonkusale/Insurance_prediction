import streamlit as st
from src.prediction import Insurance_prediction 

st.title("Insurance Prediction")
st.write("Project Description")

Age=st.number_input("Enter age : ")
Annual_Income_LPA=st.number_input("Enter annual income :")
Policy_Term_Years=st.number_input("Enter policy term years :")
Sum_Assured_Lakhs=st.number_input("Enter sum assured in lakhs :")

if st.button("Predict"):
    model=Insurance_prediction()
    result=model.prediction(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(result)
