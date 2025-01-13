import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from AppClass import RegressionLayout, RegressionModels

file_path = "clean_data.csv"
target_feature = "Life_Expectancy"

regression_model = RegressionModels(path=file_path, target=target_feature)

layout = RegressionLayout("Model Evaluation")
layout.run()

if layout.selected_regression == "Linear Regression":
    result = regression_model.linear_regression()
    st.write(f"Mean Squared Error: {result[1]}")
    st.write(f"R Squared: {result[2]}")
    st.write("### Actual vs Predicted Values")
    fig = regression_model.plot_actual_vs_predicted(regression_model.y_test, result[3])
    st.pyplot(fig)

elif layout.selected_regression == "Lasso Regression":
    result = regression_model.lasso_regression()
    st.write(f"Mean Squared Error: {result[1]}")
    st.write(f"R Squared: {result[2]}")
    st.write("### Actual vs Predicted Values")
    fig = regression_model.plot_actual_vs_predicted(regression_model.y_test, result[3])
    st.pyplot(fig)

elif layout.selected_regression == "Ridge Regression":
    result = regression_model.ridge_regression()
    st.write(f"Mean Squared Error: {result[1]}")
    st.write(f"R Squared: {result[2]}")
    st.write("### Actual vs Predicted Values")
    fig = regression_model.plot_actual_vs_predicted(regression_model.y_test, result[3])
    st.pyplot(fig)

elif layout.selected_regression == "Random Forest Regression":
    result = regression_model.random_forest_regression()
    st.write(f"Mean Squared Error: {result[1]}")
    st.write(f"R Squared: {result[2]}")
    st.write("### Actual vs Predicted Values")
    fig = regression_model.plot_actual_vs_predicted(regression_model.y_test, result[3])
    st.pyplot(fig)


# df = pd.read_csv(file_path)
# target_column = "Life_Expectancy"
#
# X = df.drop(columns=target_column)
# y = df[target_column]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
# st.write(f"**R² Score:** {r2:.2f}")
#
# # Plotting actual vs predicted values
# st.write("### Actual vs Predicted Values")
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(y_test, y_pred, color="blue", alpha=0.6)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
# ax.set_xlabel("Actual Values")
# ax.set_ylabel("Predicted Values")
# ax.set_title("Actual vs Predicted Values")
# st.pyplot(fig)
