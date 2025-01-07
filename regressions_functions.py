from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def lasso_regression(path, target, user_inputs):
    dataset, x, y = select_targets(path, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.fit_transform(x_test)

    model = LassoCV(cv=5, random_state=0, max_iter=10000)
    model.fit(x_train_scaled, y_train)
    lasso_best = Lasso(alpha=model.alpha_)
    lasso_best.fit(x_train_scaled, y_train)
    x_new = create_input_row(dataset, user_inputs, target)
    x_new_scaled = scaler.transform(x_new)
    prediction = model.predict(x_new_scaled)
    return f"Predicted Life Expectancy: {prediction[0]:.2f}"

def ridge_regression(path, target, user_inputs):
    dataset, x, y = select_targets(path, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    alphas = [0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(x_train, y_train)
    x_new = create_input_row(dataset, user_inputs, target)
    prediction = model.predict(x_new)
    return f"Predicted Life Expectancy: {prediction[0]:.2f}"


def random_forest_regression(path, target, user_inputs):
    dataset, x, y = select_targets(path, target)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    x_new = create_input_row(dataset, user_inputs, target)
    prediction = model.predict(x_new)
    return f"Predicted Life Expectancy: {prediction[0]:.2f}"


