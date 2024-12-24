import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("global_health.csv")

df = df.drop(columns=["CO2_Exposure_Percent", "Country_Code", "Life_Expectancy_Female",
                      "Life_Expectancy_Male", "Female_Population", "Male_Population",
                      "Suicide_Rate_Percent", "Water_Access_Percent"])

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
other_cols = df.select_dtypes(exclude=["float64", "int64"]).columns

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

st.write(df["Life_Expectancy"].isnull().sum())
st.write(df["Fertility_Rate"].isnull().sum())

column_name = st.selectbox("Select Column for Boxplot", df.columns)
fig_2, ax_2 = plt.subplots()
sns.boxplot(x=df[column_name], ax=ax_2)
st.pyplot(fig_2)
