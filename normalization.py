import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("global_health.csv")

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
