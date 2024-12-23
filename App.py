import os
import pandas as pd
import streamlit as st
from generate_fake_data import generate_fake_data, add_fake_data_to_real_data

data_file = 'global_health.csv'

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

original_data = load_data(data_file)

if 'fake_data.csv' not in os.listdir() and 'real_data_with_added_fake_data.csv' not in os.listdir():
    fake_data = generate_fake_data(original_data, 1000)
    added_fake_data = add_fake_data_to_real_data(original_data, fake_data)

    fake_data.to_csv('fake_data.csv', index=False)
    added_fake_data.to_csv('real_data_with_added_fake_data.csv', index=False)
