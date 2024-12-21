import pandas as pd
import numpy as np
import random

data_file = 'global_health.csv'
df = pd.read_csv(data_file)
pd.set_option('display.width', None)  # Adjust width automatically

# print(df.head())
print(df.describe())

country_code_pairs = [[country, code] for code, country in zip(df['Country_Code'].unique(), df['Country'].unique())]

years = np.arange(2012, 2022)

columns_to_randomize = [col for col in df.columns if col not in ['Year', 'Country', 'Country_Code']]
print(columns_to_randomize)

print(df["Fertility_Rate"].min())

def generate_fake_row():
    fake_data = {}

    country, code = random.choice(country_code_pairs)

    fake_data["Country"] = country
    fake_data["Country_Code"] = code
    fake_data["Year"] = float(random.choice(np.arange(2012, 2022)))

    for column in columns_to_randomize:
        min_val = df[column].min()
        max_val = df[column].max()

        precision = max(
            len(str(min_val).split('.')[1]) if '.' in str(min_val) else 0,
            len(str(max_val).split('.')[1]) if '.' in str(max_val) else 0
        )

        random_value = float(round(random.uniform(min_val, max_val), precision))

        fake_data[column] = random_value

    return fake_data

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = load_data(data_file)

num_new_rows = 5  # 5 fake data rows
new_rows = [generate_fake_row() for i in range(num_new_rows)]  # Generate 5 new rows

for row in new_rows:
    print(row)
