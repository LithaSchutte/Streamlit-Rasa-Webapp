import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path_original = "global_health.csv"
df_original = pd.read_csv(file_path_original)

model_option = ["Clean Data", "Normalized Data"]

selected_model = st.radio("Select dataset:", model_option, horizontal=True)

clean = selected_model == "Clean Data"
normalized = selected_model == "Normalized Data"

file_path = ""

if clean:
    file_path = "clean_data.csv"
elif normalized:
    file_path = "normalized.csv"

df = pd.read_csv(file_path)

target_column = "Life_Expectancy"

X = df.drop(columns=target_column)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

Fertility_Rate = st.slider("Fertility Rate", 0.00, 8.00, 2.70)
Urban_Population_Percent = st.slider("Urban Population Percent", 0.00, 100.00, 58.00)
GDP_Per_Capita = st.slider("GDP Per Capita", 100.00, 300000.00, 16000.00)
Air_Pollution = st.slider("Air Pollution", 0.00, 120.00, 30.00)
Infant_Deaths = st.slider("Infant Deaths", 0.00, 500.00, 100.00)
Immunization_Rate = st.slider("Immunization Rate", 0.00, 100.00, 85.00)


user_inputs = {"Fertility_Rate": Fertility_Rate,
               "Urban_Population_Percent": Urban_Population_Percent,
               "GDP_Per_Capita": GDP_Per_Capita,
               "Air_Pollution": Air_Pollution,
               "Infant_Deaths": Infant_Deaths,
               "Immunization_Rate": Immunization_Rate,}


def create_input_row(df, user_inputs):
    input_row = df.drop(columns=target_column).mean().to_dict()

    for key, value in user_inputs.items():
        if key in input_row:
            input_row[key] = value

    return pd.DataFrame([input_row])



X_new = create_input_row(df, user_inputs)

prediction = model.predict(X_new)


st.write("### Predicted Life Expectancy:")
st.write(f"{prediction[0]:.2f} years")

for column in X.columns:
    fig, ax = plt.subplots()
    df.sort_values(by=[column], inplace=True)
    df.set_index(column)
    ax.scatter(df[column], df["Life_Expectancy"], s=1)
    ax.plot(Urban_Population_Percent, prediction, marker="*", markersize=5, c="red")
    st.pyplot(fig)
