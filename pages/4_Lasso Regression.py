import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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



model = LassoCV(cv=5, random_state=0, max_iter=10000)

model.fit(X_train, y_train)

st.write(model.alpha_)

lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, y_train)

st.write('R squared training set', round(lasso_best.score(X_train, y_train)*100, 2))
st.write('R squared test set', round(lasso_best.score(X_test, y_test)*100, 2))

st.write(mean_squared_error(y_test, lasso_best.predict(X_test)))

y_pred = lasso_best.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Actual Life Expectancy")
ax.set_ylabel("Predicted Life Expectancy")
ax.set_title("Actual vs Predicted Life Expectancy")

st.pyplot(fig)

st.write(model.coef_)