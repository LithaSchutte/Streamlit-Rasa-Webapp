import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    columns_to_drop = [
        "CO2_Exposure_Percent", "Country_Code", "Life_Expectancy_Female", "Life_Expectancy_Male",
        "Female_Population", "Male_Population", "Suicide_Rate_Percent", "Water_Access_Percent"
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    imputer = KNNImputer(n_neighbors=5)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[[col]] = imputer.fit_transform(df[[col]])

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    correlation_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
        square=True, annot_kws={"size": 8}
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Correlation Heatmap", fontsize=14)
    return fig

def plot_boxplot(df, column_name):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[column_name], ax=ax)
    plt.title(f"Boxplot of {column_name}")
    return fig

def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, mse, r2

def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
    ax.set_xlabel("Actual Life Expectancy")
    ax.set_ylabel("Predicted Life Expectancy")
    ax.set_title("Actual vs Predicted Life Expectancy")
    ax.legend()
    return fig

"""
st.title("Global Health Data Analysis")

filepath = "global_health.csv"
df = load_data(filepath)
df = clean_data(df)
df = normalize_data(df)

st.subheader("Cleaned and Normalized Data")
st.write(df)

st.subheader("Correlation Heatmap")
heatmap_fig = plot_correlation_heatmap(df)
st.pyplot(heatmap_fig)

st.subheader("Boxplot")
column_name = st.selectbox("Select Column for Boxplot", df.columns)
boxplot_fig = plot_boxplot(df, column_name)
st.pyplot(boxplot_fig)

st.subheader("Life Expectancy Prediction")
target_column = "Life_Expectancy"
model, X_test, y_test, y_pred, mse, r2 = train_model(df, target_column)

st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R^2 Score: {r2:.2f}")

prediction_fig = plot_actual_vs_predicted(y_test, y_pred)
st.pyplot(prediction_fig)
"""