import streamlit as st
from AppClass import RegressionLayout, RegressionModels

file_path = "clean_data.csv"
target_feature = "Life_Expectancy"

layout = RegressionLayout()
layout.run()


st.write(
    "Below you can adjust some features to make a prediction. The sliders and input fields allow you to customize values"
    " for key factors that influence the prediction. Use these controls to simulate different scenarios and understand "
    "how changes in these variables affect the predicted outcomes.")

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

user_inputs = {
    "Fertility_Rate": Fertility_Rate,
    "Urban_Population_Percent": Urban_Population_Percent,
    "GDP_Per_Capita": GDP_Per_Capita,
    "Air_Pollution": Air_Pollution,
    "Infant_Deaths": Infant_Deaths,
    "Immunization_Rate": Immunization_Rate,
}

regression_model = RegressionModels(file_path, target_feature)


with output1:
    if layout.selected_regression == "Linear Regression":
        st.write(regression_model.linear_regression(user_inputs)[0])
    elif layout.selected_regression == "Lasso Regression":
        st.write(regression_model.lasso_regression(user_inputs)[0])
    elif layout.selected_regression == "Ridge Regression":
        st.write(regression_model.ridge_regression(user_inputs)[0])
    elif layout.selected_regression == "Random Forest Regression":
        st.write(regression_model.random_forest_regression(user_inputs)[0])

with output2:
    st.write("Your predicted value for life expectancy is ")


options = [
    "Immunization Rate",
    "Fertility Rate",
    "GDP Per Capita",
    "Air Pollution",
    "Infant Deaths",
    "Urban Population Percent",
]

selected_option = st.selectbox(
    'Select a feature against which to plot the target feature',
    options=options,
    index=options.index("Immunization Rate")
)
