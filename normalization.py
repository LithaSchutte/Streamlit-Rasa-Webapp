import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

imputer = KNNImputer(n_neighbors=5)

df = pd.read_csv("global_health.csv")

df = df.drop(columns=["CO2_Exposure_Percent", "Country_Code", "Life_Expectancy_Female",
                      "Life_Expectancy_Male", "Female_Population", "Male_Population",
                      "Suicide_Rate_Percent", "Water_Access_Percent"])

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
other_cols = df.select_dtypes(exclude=["float64", "int64"]).columns

for column in numeric_cols:
    st.write(f"{column} missing values: {df[column].isnull().sum()}")

df[["Hospital_Beds_Per_1000"]] = imputer.fit_transform(df[["Hospital_Beds_Per_1000"]])
df[["Alcohol_Consumption_Per_Capita"]] = imputer.fit_transform(df[["Alcohol_Consumption_Per_Capita"]])
df[["Obesity_Rate_Percent"]] = imputer.fit_transform(df[["Obesity_Rate_Percent"]])
df[["Underweight_Rate_Percent"]] = imputer.fit_transform(df[["Underweight_Rate_Percent"]])
df[["Overweight_Rate_Percent"]] = imputer.fit_transform(df[["Overweight_Rate_Percent"]])
df["Unemployment_Rate"] = df["Unemployment_Rate"].fillna(df["Unemployment_Rate"].mean())
df["Fertility_Rate"] = df["Fertility_Rate"].fillna(df["Fertility_Rate"].mean())
df["Sanitary_Expense_Per_GDP"] = df["Sanitary_Expense_Per_GDP"].fillna(df["Sanitary_Expense_Per_GDP"].mean())
df["Life_Expectancy"] = df["Life_Expectancy"].fillna(df["Life_Expectancy"].mean())
df["Infant_Deaths"] = df["Infant_Deaths"].fillna(df["Infant_Deaths"].mean())
df["GDP_Per_Capita"] = df["GDP_Per_Capita"].fillna(df["GDP_Per_Capita"].mean())
df["Immunization_Rate"] = df["Immunization_Rate"].fillna(df["Immunization_Rate"].mean())
df["Sanitary_Expense_Per_Capita"] = df["Sanitary_Expense_Per_Capita"].fillna(df["Sanitary_Expense_Per_Capita"].mean())
df[["Air_Pollution"]] = imputer.fit_transform(df[["Air_Pollution"]])
df[["Labour_Force_Total"]] = imputer.fit_transform(df[["Labour_Force_Total"]])
df[["Tuberculosis_Per_100000"]] = imputer.fit_transform(df[["Tuberculosis_Per_100000"]])
df[["Safe_Water_Access_Percent"]] = imputer.fit_transform(df[["Safe_Water_Access_Percent"]])



scaler = MinMaxScaler()

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

st.title("Normalized Data")

st.write(df)

correlation_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))  # Adjust figure size
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    square=True,
    annot_kws={"size": 8}
)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.title("Correlation Heatmap", fontsize=14)

st.pyplot(fig)

column_name = st.selectbox("Select Column for Boxplot", df.columns)
fig_2, ax_2 = plt.subplots()
sns.boxplot(x=df[column_name], ax=ax_2)
st.pyplot(fig_2)

for column in numeric_cols:
    st.write(f"{column} missing values: {df[column].isnull().sum()}")

to_exclude = "Life_Expectancy"
numeric_cols = [col for col in numeric_cols if col != to_exclude]
x = df[numeric_cols]
st.write(x)
y = df["Life_Expectancy"]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(y_pred)

fig_3, ax_3 = plt.subplots(figsize=(8, 6))
ax_3.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
ax_3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
ax_3.set_xlabel("Actual Life Expectancy")
ax_3.set_ylabel("Predicted Life Expectancy")
ax_3.set_title("Actual vs Predicted Life Expectancy")
ax_3.legend()
st.pyplot(fig_3)