import pandas as pd
import numpy as np
import random
import streamlit as st
import scipy.stats as stats


data_file = "global_health.csv"  #  data file path


def count_decimal_places(num):  # helper function
    """Count the number of decimal places in a number."""
    num_str = str(num)
    if '.' in num_str:
        return len(num_str.split('.')[1].rstrip('0'))
    else:
        return 0

def generate_fake_row(df, country_code_pairs, columns_to_randomize):
    """Generate randomised, realistic fake row using information from real dataset"""
    fake_row = {}

    country, code = random.choice(country_code_pairs)

    fake_row["Country"] = country
    fake_row["Country_Code"] = code
    fake_row["Year"] = int(random.choice(np.arange(2012, 2022)))

    # respective male and female life expectancies need to be calculated before life expectancy, life expectancy is the average
    life_expectancy_female = stats.norm.rvs(
        loc=df["Life_Expectancy_Female"].mean(),
        scale=df["Life_Expectancy_Female"].std()
    )
    gap = random.uniform(4, 6)
    # assume life expectancy gap between men and women within range 4 to 6 years to ensure realistic fake data
    life_expectancy_male = life_expectancy_female - gap

    for column in columns_to_randomize:
        min_val = df[column].min()
        max_val = df[column].max()
        first_val = df[column].iloc[0]
        last_val = df[column].iloc[-1]

        # Determine the precision based on the first and last values to ensure consistency
        precision = max(
            len(str(first_val).split('.')[1]) if '.' in str(first_val) else 0,
            len(str(last_val).split('.')[1]) if '.' in str(last_val) else 0,
        )

        if column == "Life_Expectancy":
            random_value = (life_expectancy_female + life_expectancy_male) / 2  # calculate average
        elif column == "Life_Expectancy_Female":
            random_value = life_expectancy_female  # assign previously computed value
        elif column == "Life_Expectancy_Male":
            random_value = life_expectancy_male  # assign previously computed value
        else:
            if stats.shapiro(df[column])[1] > 0.05:  # Test for normality
                random_value = stats.norm.rvs(
                    loc=df[column].mean(),
                    scale=df[column].std()
                )
            else:
                random_value = random.uniform(df[column].min(), df[column].max())

        fake_row[column] = float(round(random_value, precision))

    return fake_row


def generate_fake_data(original_data: pd.DataFrame, num_new_rows=1000) -> pd.DataFrame:
    """Generate n number of fake rows and return as a dataframe"""

    country_code_pairs = [[country, code] for code, country in zip(original_data["Country_Code"].unique(), original_data["Country"].unique())]
    columns_to_randomize = [col for col in original_data.columns if col not in ["Year", "Country", "Country_Code"]]

    new_rows = [generate_fake_row(original_data, country_code_pairs, columns_to_randomize) for _ in range(num_new_rows)]
    new_data = pd.DataFrame(new_rows)

    return new_data


def add_fake_data_to_real_data(real_data: pd.DataFrame, fake_data: pd.DataFrame) -> pd.DataFrame:
    """Join fake data with real/original data"""
    fake_dataset = real_data.copy()
    fake_dataset = pd.concat([fake_dataset, fake_data], ignore_index=True)
    return fake_dataset