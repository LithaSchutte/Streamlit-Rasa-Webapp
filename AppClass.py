import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataLoader:
    def __init__(self, file_path, cache_data=True):
        self.file_path = file_path
        self.cache_data = cache_data

    def load_data(self):
        """Load data from a CSV file with caching options."""
        if self.cache_data:
            return load_with_cache(self.file_path)
        else:
            return self._load_data()

    def _load_data(self):
        """Actual data loading logic."""
        try:
            data = pd.read_csv(self.file_path)
            return data
        except FileNotFoundError:
            st.error(f"File not found: {self.file_path}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

@st.cache_data
def load_with_cache(file_path):
    """Load data with caching enabled."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


class RegressionLayout:
    def __init__(self, title="Regression Models"):
        self.title = title
        self.regression_options = [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Random Forest Regression",
        ]
        if "selected_regression" not in st.session_state:
            st.session_state.selected_regression = "Linear Regression"
        self.selected_regression = st.session_state.selected_regression

    def display_title(self):
        st.title(self.title)

    def display_options(self):
        selected = st.radio(
            "Select a regression method:",
            self.regression_options,
            index=self.regression_options.index(st.session_state.selected_regression),
            horizontal=True,
        )

        if selected != st.session_state.selected_regression:
            st.session_state.selected_regression = selected
        self.selected_regression = st.session_state.selected_regression

    def is_selected(self, regression_type):
        """Check if a specific regression type is currently selected."""
        return self.selected_regression == regression_type

    def handle_selection(self):
        if self.is_selected("Linear Regression"):
            st.write("You selected Linear Regression.")
        elif self.is_selected("Lasso Regression"):
            st.write("You selected Lasso Regression.")
        elif self.is_selected("Ridge Regression"):
            st.write("You selected Ridge Regression.")
        elif self.is_selected("Random Forest Regression"):
            st.write("You selected Random Forest Regression.")

    def run(self):
        self.display_title()
        self.display_options()
        self.handle_selection()

class RegressionModels:
    def __init__(self, path, target):
        self.dataset = pd.read_csv(path)
        self.target = target
        self.x = self.dataset.drop(columns=self.target)
        self.y = self.dataset[self.target]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.25, random_state=20
        )

    def create_input_row(self, user_inputs):
        input_row = self.dataset.drop(columns=self.target).mean().to_dict()
        for key, value in user_inputs.items():
            if key in input_row:
                input_row[key] = value
        return pd.DataFrame([input_row])

    def _train_and_predict(self, model, user_inputs=None, scaler=None):
        x_train = self.x_train
        x_test = self.x_test
        if scaler:
            scaler = scaler.fit(self.x_train)
            x_train = scaler.transform(self.x_train)
            x_test = scaler.transform(self.x_test)

        model.fit(x_train, self.y_train)

        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        mse = mean_squared_error(self.y_test, y_test_pred)
        r_squared = r2_score(self.y_test, y_test_pred)

        if user_inputs:
            x_new = self.create_input_row(user_inputs)
            if scaler:
                x_new = scaler.transform(x_new)

            prediction = model.predict(x_new)
            return [f"Predicted Life Expectancy: {prediction[0]:.2f}", mse, r_squared, y_test_pred]
        else:
            return [None, mse, r_squared, y_test_pred]

    def linear_regression(self, user_inputs=None):
        model = LinearRegression()
        return self._train_and_predict(model, user_inputs)

    def lasso_regression(self, user_inputs=None):
        scaler = StandardScaler()
        lasso_cv_model = LassoCV(cv=5, random_state=0, max_iter=10000)
        lasso_cv_model.fit(scaler.fit_transform(self.x_train), self.y_train)

        best_model = Lasso(alpha=lasso_cv_model.alpha_)
        return self._train_and_predict(best_model, user_inputs, scaler=scaler)

    def ridge_regression(self, user_inputs=None):
        scaler = MinMaxScaler()
        alphas = [0.1, 1.0, 10.0, 100.0]
        model = RidgeCV(alphas=alphas, cv=5)
        return self._train_and_predict(model, user_inputs, scaler=scaler)

    def random_forest_regression(self, user_inputs=None):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        return self._train_and_predict(model, user_inputs)

    def plot_actual_vs_predicted(self, y_test, y_pred):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='blue', label="Predicted")
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color='red',
            linestyle='--',
            label="Perfect Prediction"
        )
        ax.set_title("Actual vs Predicted Values")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        ax.grid(True)
        return fig
