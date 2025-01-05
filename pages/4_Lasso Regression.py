import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file_path_original = "global_health.csv"
df_original = pd.read_csv(file_path_original)
