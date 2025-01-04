import streamlit as st
from normalization import (load_data, clean_data, normalize_data, train_model, plot_actual_vs_predicted)

st.title("Regression Models")

regression_options = ["Linear Regression", "Lasso Regression"]

selected_regression = st.radio(
    "Select a regression method:",
    regression_options,
    horizontal=True)

is_linear_regression = selected_regression == "Linear Regression"
is_lasso_regression = selected_regression == "Lasso Regression"

col1, col2 = st.columns([1, 2])


with col1:
    options = ["Original Data", "Data with Added Synthetic Data"]

    selected_option = st.radio(
        "Select a dataset to do predictions with:",
        options,
        horizontal=True)

    file_mapping = {
        "Original Data": "global_health.csv",
        "Data with Added Synthetic Data": "real_data_with_added_fake_data.csv"
    }

    data_file = file_mapping.get(selected_option)

    year = st.slider("Select a year", 2012, 2021, 2015)

    Fertility_Rate = st.slider("Fertility Rate", 0.00, 8.00, 2.70)
    Urban_Population_Percent = st.slider("Urban Population Percent", 0.00, 100.00, 58.00)
    Overweight_Rate_Percent = st.slider("Overweight Rate Percent", 0.00, 100.00, 42.00)
    Sanitary_Expense_Per_GDP = st.slider("Sanitary Expense per GDP", 0.00, 12000.00, 1000.00)

with col2:
    df = load_data(data_file)
    df = clean_data(df)
    df = normalize_data(df)

    st.subheader("Life Expectancy Prediction")
    target_column = "Life_Expectancy"
    model, X_test, y_test, y_pred, mse, r2 = train_model(df, target_column)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

    prediction_fig = plot_actual_vs_predicted(y_test, y_pred)
    st.pyplot(prediction_fig)