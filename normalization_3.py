import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Linear Regression Visualization")

file_path = "global_health.csv"

if file_path:
    try:
        data = pd.read_csv(file_path)
        st.write("Dataset Preview:")
        st.write(data.head())

        default_cols_to_drop = [
        "CO2_Exposure_Percent", "Country_Code", "Life_Expectancy_Female", "Life_Expectancy_Male",
        "Female_Population", "Male_Population", "Suicide_Rate_Percent", "Water_Access_Percent", "Country"
        ]

        st.write("Select columns to drop (if any):")
        columns = data.columns.tolist()
        cols_to_drop = st.multiselect("Columns to drop:", columns, default=default_cols_to_drop)

        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            st.write("Updated Dataset Preview:")
            st.write(data.head())

        fill_method = st.radio("Select method to fill missing values:", ("Mean", "KNN"))

        num_cols = data.select_dtypes(include= [np.number]).columns

        if fill_method == "Mean":
            data[num_cols] = data[num_cols].apply(lambda col: col.fillna(col.mean()))
        elif fill_method == "KNN":
            knn_imputer = KNNImputer(n_neighbors=15)
            data[num_cols] = knn_imputer.fit_transform(data[num_cols])

        columns = data.columns.tolist()
        target_col = st.selectbox("Select Target Column (Y):", columns, index=columns.index("Life_Expectancy"))

        if target_col:
            X = data.drop(columns=[target_col]).values
            y = data[target_col].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = LinearRegression()
            model.fit(X, y)

            y_pred = model.predict(X)

            # Step 9: Plot the relationship between actual and predicted values
            plt.figure(figsize=(8, 6))
            plt.scatter(y, y_pred, alpha=0.7, label="Actual vs Predicted")
            plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label="Perfect Fit")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Linear Regression: {target_col} vs Features")
            plt.legend()
            st.pyplot(plt)


    except Exception as e:
        st.error(f"Error reading the file: {e}")