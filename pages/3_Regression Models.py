import streamlit as st
from matplotlib import pyplot as plt


from AppClass import RegressionLayout, RegressionModels, DataLoader

file_path = "clean_data.csv"
target_feature = "Life_Expectancy"
data_loader = DataLoader("global_health.csv", cache_data=True)
df = data_loader.load_data()
layout = RegressionLayout()
layout.run()


st.write(
    "Below you can adjust some features to make a prediction. The sliders and input fields allow you to customize values"
    " for key factors that influence the prediction. Use these controls to simulate different scenarios and understand "
    "how changes in these variables affect the predicted outcomes.")

col1, col2, col3 = st.columns(3)

with col1:
    Fertility_Rate = st.slider("Fertility Rate", 0.00, 8.00, 2.70)
    Sanitary_Expense_Per_Capita = st.slider("Sanitary Expense Per Capita", 10.00, 15000.00, 1160.00)

with col2:
    Air_Pollution = st.slider("Air Pollution", 0.00, 120.00, 30.00)
    Safe_Water_Access_Percent = st.slider("Safe Water Access Percent", 0, 100, 85)

with col3:
    Urban_Population_Percent = st.number_input("Urban Population Percent", min_value=0, max_value=100, value=58, step=1)
    Immunization_Rate = st.number_input("Immunization Rate", min_value=0, max_value=100, value=85, step=1)

user_inputs = {
    "Fertility_Rate": Fertility_Rate,
    "Urban_Population_Percent": Urban_Population_Percent,
    "Sanitary_Expense_Per_Capita": Sanitary_Expense_Per_Capita,
    "Air_Pollution": Air_Pollution,
    "Safe_Water_Access_Percent": Safe_Water_Access_Percent,
    "Immunization_Rate": Immunization_Rate,
}

regression_model = RegressionModels(file_path, target_feature)

output1, output2 = st.columns(2)

with output1:
    if layout.selected_regression == "Linear Regression":
        prediction = regression_model.linear_regression(user_inputs)[0]
        st.write(f"Predicted Life Expectancy: {prediction:.2f}")
    elif layout.selected_regression == "Lasso Regression":
        prediction = regression_model.lasso_regression(user_inputs)[0]
        st.write(f"Predicted Life Expectancy: {prediction:.2f}")
    elif layout.selected_regression == "Ridge Regression":
        prediction = regression_model.ridge_regression(user_inputs)[0]
        st.write(f"Predicted Life Expectancy: {prediction:.2f}")
    elif layout.selected_regression == "Random Forest Regression":
        prediction = regression_model.random_forest_regression(user_inputs)[0]
        st.write(f"Predicted Life Expectancy: {prediction:.2f}")


with output2:
    st.write("How your prediction compares to the mean:")
    st.write(f"The mean value for life expectancy is {round(df[target_feature].mean(), 2)}")
    st.write(f"The mean value for female life expectancy is {round(df['Life_Expectancy_Female'].mean(), 2)}")
    st.write(f"The mean value for male life expectancy is {round(df['Life_Expectancy_Male'].mean(), 2)}")

options = [
    "Immunization_Rate",
    "Fertility_Rate",
    "Sanitary_Expense_Per_Capita",
    "Air_Pollution",
    "Safe_Water_Access_Percent",
    "Urban_Population_Percent",
]

selected_column = st.selectbox(
    'Select a feature against which to plot the target feature',
    options=options,
    index=options.index("Immunization_Rate")
)


fig, ax = plt.subplots()
df.sort_values(by=[selected_column], inplace=True)

# Update x-axis values based on the selected column
ax.scatter(df[selected_column], df["Life_Expectancy"], s=10)

# Update the prediction line plot for the selected column, based on the value for that column
ax.plot(user_inputs[selected_column], prediction, marker="*", markersize=5, c="red")

# Set x-axis limits based on the selected column range
ax.set_xlim(df[selected_column].min(), df[selected_column].max())
ax.set_xlabel(selected_column)
ax.set_ylabel("Life Expectancy")
st.pyplot(fig)
