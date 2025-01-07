import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

file_path = "clean_data.csv"
df = pd.read_csv(file_path)

target_column = "Life_Expectancy"

y = df[target_column]
X = df.drop(columns=target_column)

list_numerical = X.columns
st.write(list_numerical)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler().fit(X_train[list_numerical])
X_train[list_numerical] = scaler.transform(X_train[list_numerical])
X_test[list_numerical] = scaler.transform(X_test[list_numerical])



alphas = [0.1, 1.0, 10.0, 100.0]

model = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)


r_squared = model.score(X_test, y_test)
st.write("R-squared:", r_squared)

y_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Actual Life Expectancy")
ax.set_ylabel("Predicted Life Expectancy")
ax.set_title("Actual vs Predicted Life Expectancy")

st.pyplot(fig)

st.write(model.coef_)