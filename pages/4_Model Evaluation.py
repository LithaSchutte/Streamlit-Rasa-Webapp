import streamlit as st
from AppClass import RegressionLayout, RegressionModels

file_path = "clean_data.csv"
target_feature = "Life_Expectancy"

regression_model = RegressionModels(path=file_path, target=target_feature)

layout = RegressionLayout("Model Evaluation")
layout.run()

regression_methods = {
    "Linear Regression": regression_model.linear_regression,
    "Lasso Regression": regression_model.lasso_regression,
    "Ridge Regression": regression_model.ridge_regression,
    "Random Forest Regression": regression_model.random_forest_regression
}

if layout.selected_regression in regression_methods:
    result = regression_methods[layout.selected_regression]()
    st.write(f"Mean Squared Error: {result[1]:.2f}")
    st.write(f"R Squared: {result[2]:.2f}")
    st.write("### Actual vs Predicted Values")
    fig = regression_model.plot_actual_vs_predicted(regression_model.y_test, result[3])
    st.pyplot(fig)