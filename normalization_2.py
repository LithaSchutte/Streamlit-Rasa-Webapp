import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df_original = pd.read_csv("global_health.csv")

to_drop = [
        "CO2_Exposure_Percent", "Country_Code", "Life_Expectancy_Female", "Life_Expectancy_Male",
        "Female_Population", "Male_Population", "Suicide_Rate_Percent", "Water_Access_Percent"
        ]

numerical_cols = df_original.select_dtypes(include=["float64", "int64"]).columns.difference(to_drop)

def drop_columns(dataset, columns):
    dataset = dataset.drop(columns=columns, errors="ignore")
    return dataset

def fill_missing_values(dataset, columns):
    imputer = KNNImputer(n_neighbors=5)
    for col in columns:
        if dataset[col].isnull().sum() > 0:
            dataset[[col]] = imputer.fit_transform(dataset[[col]])
    return dataset

def normalize(dataset, columns):
    scaler = MinMaxScaler()
    dataset[columns] = scaler.fit_transform(dataset[columns])
    return dataset

df_modified = fill_missing_values(df_original, numerical_cols)
df_modified = drop_columns(df_modified, to_drop)

col1, col2 = st.columns(2)

with col1:
    st.write("Original Dataset")
    st.write(df_original)
    column = "Year"
    min_value = df_modified[column].min()
    max_value = df_modified[column].max()
    selected_year = st.slider(f"Select a value for {column}",
                              min_value=int(min_value),
                              max_value=int(max_value),
                              value=int(min_value))

    column = "Fertility_Rate"
    min_value = df_modified[column].min()
    max_value = df_modified[column].max()
    selected_fertility = st.slider(f"Select a value for {column}",
                                   min_value=float(min_value),
                                   max_value=float(max_value),
                                   value=float(min_value))

    column = "Total_Population"
    min_value = df_modified[column].min()
    max_value = df_modified[column].max()
    selected_population = st.slider(f"Select a value for {column}",
                                   min_value=float(min_value),
                                   max_value=float(max_value),
                                   value=float(min_value))

with col2:
    st.write("No missing values and dropped columns")
    st.write(df_modified)

    target_column = "Life_Expectancy"

    X = df_modified.drop(columns=[target_column, "Country"])
    y = df_modified[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    user_input = {"Year": selected_year,
                  "Fertility_Rate": selected_fertility,
                  "Total_Population": selected_population
                  }

    input_df = pd.DataFrame(user_input, index=[0])

    input_df = pd.get_dummies(input_df, drop_first=True)

    missing_cols = set(X_test.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[X_test.columns]

    user_prediction = model.predict(input_df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predictions")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal Fit")

    user_actual = user_prediction[0]
    ax.scatter(user_actual, user_actual, color="green", label="User Prediction", s=100, zorder=5)

    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    ax.set_title("Actual vs Predicted Life Expectancy")
    ax.legend()

    st.pyplot(fig)

    st.write(f"Predicted Life Expectancy: {user_prediction[0]:.2f}")