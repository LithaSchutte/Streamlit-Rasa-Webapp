from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# file_path = "clean_data.csv"

def create_input_row(df, user_inputs, target):
    input_row = df.drop(columns=target).mean().to_dict()
    for key, value in user_inputs.items():
        if key in input_row:
            input_row[key] = value

    return pd.DataFrame([input_row])

def select_targets(path, target):
    dataset = pd.read_csv(path)
    x = dataset.drop(columns=target)
    y = dataset[target]
    return dataset, x, y

def linear_regression(path, target, user_inputs):
    dataset, x, y = select_targets(path, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    x_new = create_input_row(dataset, user_inputs, target)
    prediction = model.predict(x_new)
    return f"Predicted Life Expectancy: {prediction[0]:.2f}"

def lasso_regression():
    pass

def ridge_regression():
    pass

def random_forest_regression():
    pass