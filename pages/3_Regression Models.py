import streamlit as st
from AppClass import RegressionLayout
from regressions_functions import linear_regression, lasso_regression, random_forest_regression, ridge_regression

file_path = "clean_data.csv"

layout = RegressionLayout()
layout.run()

# Input sliders for regression features
col1, col2, col3 = st.columns(3)

with col1:
    Fertility_Rate = st.slider("Fertility Rate", 0.00, 8.00, 2.70)
    GDP_Per_Capita = st.slider("GDP Per Capita", 100.00, 300000.00, 16000.00)

with col2:
    Air_Pollution = st.slider("Air Pollution", 0.00, 120.00, 30.00)
    Infant_Deaths = st.slider("Infant Deaths", 0.00, 500.00, 100.00)

with col3:
    Urban_Population_Percent = st.number_input("Urban Population Percent", min_value=0, max_value=100, value=58, step=1)
    Immunization_Rate = st.number_input("Immunization Rate", min_value=0, max_value=100, value=85, step=1)

user_inputs = {"Fertility_Rate": Fertility_Rate,
               "Urban_Population_Percent": Urban_Population_Percent,
               "GDP_Per_Capita": GDP_Per_Capita,
               "Air_Pollution": Air_Pollution,
               "Infant_Deaths": Infant_Deaths,
               "Immunization_Rate": Immunization_Rate,}


if layout.selected_regression == "Linear Regression":
    st.write(linear_regression(file_path, "Life_Expectancy", user_inputs))
elif layout.selected_regression == "Lasso Regression":
    st.write(lasso_regression(file_path, "Life_Expectancy", user_inputs))
elif layout.selected_regression == "Ridge Regression":
    st.write(ridge_regression(file_path, "Life_Expectancy", user_inputs))
elif layout.selected_regression == "Random Forest Regression":
    st.write(random_forest_regression(file_path, "Life_Expectancy", user_inputs))

options = ["Immunization Rate", "Fertility Rate", "GDP Per Capita", "Air Pollution", "Infant Deaths", "Urban Population Percent"]

selected_option = st.selectbox(
    'Select a feature against which to plot the target feature',
    options=options,
    index=options.index("Immunization Rate")
)
